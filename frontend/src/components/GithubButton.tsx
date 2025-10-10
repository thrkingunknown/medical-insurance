import { Github } from "lucide-react";
import { Button } from "./ui/button";

export function GithubButton() {
  return (
    <Button
      variant="outline"
      size="icon"
      onClick={() =>
        window.open(
          "https://github.com/thrkingunknown/medical-insurance",
          "_blank"
        )
      }
      className="rounded-full w-10 h-10 transition-all duration-300 hover:scale-110 border-2 dark:border-gray-700"
      aria-label="View on GitHub"
    >
      <Github className="h-[1.2rem] w-[1.2rem] transition-all" />
    </Button>
  );
}
