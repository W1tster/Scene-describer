from collections import Counter


class SceneDescriber:
    def describe(self, objects):
        if not objects:
            return "The scene appears to be empty."

        counts = Counter(objects)
        parts = []

        for obj_name, count in counts.items():
            if count == 1:
                parts.append(f"a {obj_name}")
            else:
                # basic pluralization — handles most english words well enough
                if obj_name.endswith(('s', 'x', 'ch', 'sh')):
                    plural = f"{obj_name}es"
                else:
                    plural = f"{obj_name}s"
                parts.append(f"{count} {plural}")

        if len(parts) == 1:
            return f"The scene contains {parts[0]}."
        elif len(parts) == 2:
            return f"The scene contains {parts[0]} and {parts[1]}."
        else:
            return f"The scene contains {', '.join(parts[:-1])}, and {parts[-1]}."


if __name__ == "__main__":
    d = SceneDescriber()
    print(d.describe([]))
    print(d.describe(["dog"]))
    print(d.describe(["person", "dog"]))
    print(d.describe(["person", "person", "dog", "car", "car"]))
