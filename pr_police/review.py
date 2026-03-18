import os
import requests
import sys
import re


def get_diff() -> str:
    """
    Reads the git diff from pr_diff.txt.
    Exits cleanly if the file is missing or the diff is empty.

    Returns:
        str: The raw git diff content.
    """
    try:
        with open("pr_police/pr_diff.txt", "r") as f:
            diff = f.read()
        if not diff.strip():
            print("No diff found, skipping.")
            sys.exit(0)
        return diff
    except FileNotFoundError:
        print("pr_diff.txt not found")
        sys.exit(1)

def sanitize_prompt(prompt: str) -> str:
    """
    Sanitizes a prompt before passing it to the model.
    Removes null bytes, strips excessive whitespace, truncates to a
    safe token limit, and blocks prompt injection attempts.

    Args:
        prompt (str): The prompt to send to the model.

    Returns:
        str: The sanitized prompt.

    Raises:
        ValueError: If the prompt is empty after sanitization.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty")

    # Remove null bytes and non-printable characters
    prompt = prompt.replace("\x00", "")
    prompt = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', '', prompt)

    # Collapse excessive blank lines (more than 2 in a row)
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)

    # Strip leading/trailing whitespace
    prompt = prompt.strip()

    # Block prompt injection attempts
    injection_patterns = [
        r'ignore (all |previous |above )?instructions',
        r'disregard (all |previous |above )?instructions',
        r'you are now',
        r'new persona',
        r'forget (all |everything|your )?(you |previously )?know',
        r'system prompt',
        r'jailbreak',
    ]
    for pattern in injection_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            print(pattern)
            print(f"Potential prompt injection detected, neutralizing...")
            prompt = re.sub(pattern, '[REDACTED]', prompt, flags=re.IGNORECASE)

    return prompt

def ask_model(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama model via the FastAPI endpoint
    and returns the model's response.

    Args:
        prompt (str): The prompt to send to the model.

    Returns:
        str: The model's response text.

    Raises:
        ValueError: If PR_REVIEW_URL is not set or the model returns an empty response.
        requests.RequestException: If the request to the model fails.
    """
    pr_review_url = os.getenv('PR_REVIEW_URL', 'http://localhost:8000/generate')
    model = os.getenv('MODEL') or 'qwen2.5-coder:7b'

    if not pr_review_url:
        raise ValueError("PR_REVIEW_URL env var is not set")

    try:
        response = requests.post(
            pr_review_url, json={'prompt': sanitize_prompt(prompt), 'model': model}, timeout=999999)
        response.raise_for_status()
        review = response.json().get('response', '')

        if not review:
            raise ValueError("No review received from the server")
        return review
    except requests.RequestException as e:
        print(f"Error making request to model: {e}")
        raise


def get_review(diff: str) -> str:
    """
    Builds the code review prompt and sends it to the model.
    The prompt instructs the model to return a verdict, star ratings,
    code quality assessment, and structured inline comments.

    Args:
        diff (str): The raw git diff to review.

    Returns:
        str: The full review text including verdict and inline comment markers.
    """
    prompt = f"""You are a senior code reviewer.
        Your tone should be conversational, as if you're a cool older guy
        mentoring a younger junior developer.
        Make a joke about how old you are.

        At the top of the file: put VERDICT: CODE IS REJECTED
        if there is an immediate security risk.
        Otherwise, put VERDICT: CODE IS CONDITIONALLY ACCEPTED.

        Review this git diff and assess:
        1. PEP-8 Compliance
        2. Possible bugs
        3. Possible security considerations

        Include the suggested code changes.

        If applicable, generate simple unit tests for all the functions covered in the diff.

        After your review, add a section called INLINE COMMENTS
        in this exact format, one per line:
        INLINE::filename.py::42::Your comment about this specific line

        Diff:
        {diff}"""
    return ask_model(prompt)


def get_pr_context() -> tuple[str, str, str, str]:
    """
    Retrieves GitHub context from environment variables and the GitHub API.
    Fetches the token, repository name, PR number, and latest commit SHA.

    Returns:
        tuple[str, str, str, str]: (gh_token, repo, pr_number, commit_sha)

    Raises:
        KeyError: If any required environment variable is missing.
    """
    gh_token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]

    try:
        pr_data = requests.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
            headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"}
        ).json()
        commit_sha = pr_data["head"]["sha"]
        print(f"Latest commit SHA: {commit_sha}")
    except:
        print("Failed to get PR context")
        return None, None, None, None

    return gh_token, repo, pr_number, commit_sha


def post_review_comment(gh_token: str, repo: str, pr_number: str, review: str) -> None:
    """
    Posts the main review as a PR comment, stripping out the raw
    INLINE:: markers so the comment body is clean and readable.

    Args:
        gh_token (str): GitHub personal access token.
        repo (str): Repository in the format 'owner/repo'.
        pr_number (str): The PR number to comment on.
        review (str): The full review text from the model.
    """
    review_body = review.split("INLINE COMMENTS")[0].strip() if "INLINE COMMENTS" in review else review

    result = requests.post(
        f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments",
        headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"},
        json={"body": f"## 🚔 PR Police Report\n\n{review_body}"}
    )

    print(f"GitHub API status: {result.status_code}")
    if result.status_code == 201:
        print("Posted main review successfully")
    else:
        print(f"Failed to post review: {result.text}")
        sys.exit(1)


def post_inline_comments(gh_token: str, repo: str, pr_number: str, commit_sha: str, review: str) -> None:
    """
    Parses INLINE:: markers from the model's review and posts each one
    as an inline comment on the relevant file and line in the PR diff.
    Failures on individual comments are non-fatal — a warning is printed
    and the run continues.

    Args:
        gh_token (str): GitHub personal access token.
        repo (str): Repository in the format 'owner/repo'.
        pr_number (str): The PR number to comment on.
        commit_sha (str): The latest commit SHA on the PR branch.
        review (str): The full review text containing INLINE:: markers.
    """
    inline_pattern = re.compile(r'INLINE::(.+?)::(\d+)::(.+)', re.IGNORECASE)
    inline_matches = inline_pattern.findall(review)

    if not inline_matches:
        print("No inline comments found in review.")
        return

    print(f"Found {len(inline_matches)} inline comments, posting...")
    for filename, line_num, comment in inline_matches:
        result = requests.post(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments",
            headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"},
            json={
                "body": f"🚔 {comment.strip()}",
                "commit_id": commit_sha,
                "path": filename.strip(),
                "line": int(line_num),
                "side": "RIGHT"
            }
        )
        if result.status_code == 201:
            print(f"Inline comment posted on {filename}:{line_num}")
        else:
            print(f"Could not post inline on {filename}:{line_num} — {result.text}")


def populate_pr_description(gh_token: str, repo: str, pr_number: str, diff: str) -> None:
    """
    Checks if the PR has an existing description. If empty, asks the model
    to generate a concise one from the diff and patches it onto the PR.

    Args:
        gh_token (str): GitHub personal access token.
        repo (str): Repository in the format 'owner/repo'.
        pr_number (str): The PR number to update.
        diff (str): The raw git diff used to generate the description.
    """
    print("Checking PR description...")
    pr = requests.get(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"}
    ).json()

    if pr.get("body"):
        print("PR already has a description, skipping.")
        return

    generated_description = ask_model(f"Write a concise PR description for this diff:\n{diff}")
    print(generated_description)

    result = requests.patch(
        f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
        headers={"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"},
        json={"body": generated_description}
    )

    if result.status_code == 200:
        print("PR description updated successfully")
    else:
        print(f"Failed to update description: {result.text}")


def check_verdict(review: str) -> None:
    """
    Reads the first line of the model's review to determine the verdict.
    Exits with code 1 (fails the GitHub Action) if the code is rejected,
    or code 0 (passes) if conditionally accepted.

    Args:
        review (str): The full review text from the model.
    """
    first_line = review.upper().split("\n")[0]
    if "VERDICT: CODE IS REJECTED" in first_line:
        print("Code has been rejected")
        sys.exit(1)
    else:
        print("Code has been accepted")
        sys.exit(0)


if __name__ == "__main__":
    diff = get_diff()

    try:
        review = get_review(diff)
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error getting review: {e}")
        sys.exit(1)

    try:
        gh_token, repo, pr_number, commit_sha = get_pr_context()
        post_review_comment(gh_token, repo, pr_number, review)
        post_inline_comments(gh_token, repo, pr_number, commit_sha, review)
        populate_pr_description(gh_token, repo, pr_number, diff)
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        sys.exit(1)

    check_verdict(review)
