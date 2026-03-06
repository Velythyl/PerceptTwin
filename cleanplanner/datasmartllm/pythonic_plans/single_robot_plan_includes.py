def get_includes_text():
    curpath = "/".join(__file__.split("/")[:-1])
    prompt = curpath + "/single_robot_plan_prompt.py"

    lines = []
    with open(prompt, 'r') as f:
        IS_IN_DEF = False
        for line in f.readlines():
            if line.startswith("def"):
                IS_IN_DEF = True
                lines.append(line)
                continue

            if IS_IN_DEF:
                if line.startswith(" ") or line.startswith("\t"):
                    lines.append(line)
                    continue
                else:
                    IS_IN_DEF = False
                    continue

    return "\n".join(lines)

if __name__ == "__main__":
    get_includes_text()