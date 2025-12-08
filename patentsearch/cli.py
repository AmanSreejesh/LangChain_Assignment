from .pipeline import run_patentsearch_pipeline, pretty_print_result


def main():
    print("Enter invention description (end with Ctrl+D / Ctrl+Z):")
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass

    idea = "\n".join(lines).strip()
    if not idea:
        print("No idea text provided.")
    else:
        result = run_patentsearch_pipeline(idea)
        pretty_print_result(result)


if __name__ == "__main__":
    main()
