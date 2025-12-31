#!/usr/bin/env python3
"""
Meme Pack Generator using Gemini Nano Banana Pro (gemini-3-pro-image-preview)
Generates personalized memes by merging a person's face with meme templates.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict, field

from google import genai
from google.genai import types
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()

console = Console()

# Model IDs
IMAGE_MODEL = "gemini-3-pro-image-preview"  # Nano Banana Pro for image generation
ANALYSIS_MODEL = "gemini-2.0-flash"  # For analyzing templates


@dataclass
class MemeTemplate:
    """A meme template with its optimized prompt."""
    filename: str
    name: str
    prompt: str
    description: str
    expression_needed: str = "neutral"  # e.g., "laughing", "serious", "confused", "angry"
    angle_needed: str = "front"  # e.g., "front", "3quarter", "profile"


@dataclass
class ReferenceImage:
    """A reference image with metadata."""
    filename: str
    angle: str
    expression: str
    use_for: list[str] = field(default_factory=list)
    primary: bool = False


class MemeGenerator:
    """Generates personalized memes using Gemini Nano Banana Pro."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_AI_KEY not found. Set it in .env or pass it directly.")

        self.client = genai.Client(api_key=self.api_key)
        self.base_dir = Path(__file__).parent
        self.templates_dir = self.base_dir / "memes" / "templates"
        self.references_dir = self.base_dir / "reference_images"
        self.output_dir = self.base_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        self.cache_file = self.templates_dir / "prompts_cache.json"
        self.templates: dict[str, MemeTemplate] = {}
        self.references: dict[str, ReferenceImage] = {}
        self.reference_metadata: dict = {}
        self._load_cache()
        self._load_references()

    def _load_cache(self):
        """Load cached prompt analysis."""
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                data = json.load(f)
                for name, template_data in data.items():
                    # Handle both old and new format
                    if "expression_needed" not in template_data:
                        template_data["expression_needed"] = "neutral"
                    if "angle_needed" not in template_data:
                        template_data["angle_needed"] = "front"
                    self.templates[name] = MemeTemplate(**template_data)

    def _save_cache(self):
        """Save prompt analysis to cache."""
        data = {name: asdict(t) for name, t in self.templates.items()}
        with open(self.cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def _load_references(self):
        """Load reference images metadata."""
        metadata_file = self.references_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.reference_metadata = json.load(f)
                for filename, data in self.reference_metadata.get("images", {}).items():
                    self.references[filename] = ReferenceImage(
                        filename=filename,
                        angle=data.get("angle", "front"),
                        expression=data.get("expression", "neutral"),
                        use_for=data.get("use_for", []),
                        primary=data.get("primary", False),
                    )

    def _get_mime_type(self, path: str) -> str:
        suffix = Path(path).suffix.lower()
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/png")

    def select_best_reference(self, template: MemeTemplate) -> Path:
        """Select the best reference image for a given meme template."""

        if not self.references:
            # Fallback: just use first image in references dir
            refs = list(self.references_dir.glob("*.*"))
            refs = [r for r in refs if r.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]]
            if refs:
                return refs[0]
            raise ValueError("No reference images found!")

        expression = template.expression_needed.lower()
        angle = template.angle_needed.lower()

        # Score each reference image
        scores: dict[str, int] = {}

        for filename, ref in self.references.items():
            filepath = self.references_dir / filename
            if not filepath.exists():
                continue

            score = 0

            # Primary reference gets base score
            if ref.primary:
                score += 5

            # Expression matching
            ref_expr = ref.expression.lower()
            if expression in ref_expr or any(expression in u.lower() for u in ref.use_for):
                score += 10
            elif "neutral" in ref_expr and expression == "neutral":
                score += 8
            elif "smile" in ref_expr and expression in ["happy", "friendly", "smiling"]:
                score += 7
            elif "laugh" in ref_expr and expression in ["laughing", "happy", "joyful"]:
                score += 10
            elif "serious" in ref_expr and expression in ["serious", "angry", "intense"]:
                score += 8
            elif "confused" in ref_expr and expression in ["confused", "skeptical"]:
                score += 10
            elif "frustrated" in ref_expr and expression in ["frustrated", "angry", "annoyed"]:
                score += 10

            # Angle matching
            ref_angle = ref.angle.lower()
            if angle in ref_angle:
                score += 5
            elif angle == "front" and "front" in ref_angle:
                score += 5
            elif angle == "3quarter" and "3quarter" in ref_angle:
                score += 5

            # Check use_for tags
            for use_tag in ref.use_for:
                if expression in use_tag.lower() or angle in use_tag.lower():
                    score += 3

            scores[filename] = score

        # Get highest scoring reference
        if scores:
            best = max(scores, key=scores.get)
            return self.references_dir / best

        # Fallback to primary or first available
        for filename, ref in self.references.items():
            if ref.primary:
                return self.references_dir / filename

        return self.references_dir / list(self.references.keys())[0]

    def analyze_template(self, image_path: Path) -> Optional[MemeTemplate]:
        """Use Gemini to analyze a meme template and generate an optimized prompt."""

        filename = image_path.name
        if filename in self.templates:
            return self.templates[filename]

        console.print(f"[dim]Analyzing: {filename}[/dim]")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        analysis_prompt = """Analyze this meme template image. I need to create a prompt for Gemini's image generation that will recreate this meme but with a different person's face.

Provide a JSON response with:
1. "name": A short, memorable name for this meme (2-4 words)
2. "description": One sentence describing what emotion/situation this meme represents
3. "expression_needed": What facial expression should the person have? (e.g., "laughing", "serious", "confused", "smug", "angry", "neutral", "smiling", "skeptical")
4. "angle_needed": What angle is the face at? (e.g., "front", "3quarter", "profile", "looking_up", "looking_down")
5. "prompt": A detailed prompt for recreating this meme with someone else's face. The prompt should:
   - Describe the exact pose, expression, lighting, and composition
   - Specify where the person's face should go
   - Include details about clothing, background, props if visible
   - Emphasize preserving the person's exact facial features (glasses, facial hair, etc.)
   - Match the style (photo, stylized, etc.)

Return ONLY valid JSON:
{"name": "...", "description": "...", "expression_needed": "...", "angle_needed": "...", "prompt": "..."}"""

        try:
            response = self.client.models.generate_content(
                model=ANALYSIS_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=image_bytes, mime_type=self._get_mime_type(str(image_path))),
                            types.Part.from_text(text=analysis_prompt),
                        ],
                    )
                ],
            )

            response_text = response.text.strip()
            # Extract JSON
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0]
            else:
                json_str = response_text

            data = json.loads(json_str.strip())

            template = MemeTemplate(
                filename=filename,
                name=data.get("name", filename),
                prompt=data.get("prompt", ""),
                description=data.get("description", ""),
                expression_needed=data.get("expression_needed", "neutral"),
                angle_needed=data.get("angle_needed", "front"),
            )

            self.templates[filename] = template
            self._save_cache()
            return template

        except Exception as e:
            console.print(f"[red]Error analyzing {filename}: {e}[/red]")
            return None

    def analyze_all_templates(self):
        """Analyze all meme templates in the templates folder."""
        template_files = list(self.templates_dir.glob("*.png")) + list(self.templates_dir.glob("*.jpg"))

        new_templates = [f for f in template_files if f.name not in self.templates]

        if not new_templates:
            console.print(f"[green]All {len(template_files)} templates already analyzed (cached)[/green]")
            return

        console.print(f"\n[bold]Analyzing {len(new_templates)} new meme templates...[/bold]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing templates...", total=len(new_templates))

            for template_file in new_templates:
                progress.update(task, description=f"Analyzing: {template_file.name}")
                self.analyze_template(template_file)
                progress.advance(task)

        console.print(f"[green]Analyzed {len(self.templates)} templates total[/green]")

    def generate_meme(
        self,
        template_filename: str,
        person_name: str = "this person",
        output_path: Optional[Path] = None,
        reference_override: Optional[Path] = None,
    ) -> Optional[Path]:
        """Generate a single meme with the person's face."""

        template_path = self.templates_dir / template_filename
        if not template_path.exists():
            console.print(f"[red]Template not found: {template_filename}[/red]")
            return None

        # Ensure template is analyzed
        if template_filename not in self.templates:
            template = self.analyze_template(template_path)
            if not template:
                return None
        else:
            template = self.templates[template_filename]

        # Select best reference image
        if reference_override:
            reference_path = reference_override
        else:
            reference_path = self.select_best_reference(template)

        console.print(f"[dim]  Using reference: {reference_path.name} (needs: {template.expression_needed}, {template.angle_needed})[/dim]")

        # Load images
        with open(reference_path, "rb") as f:
            person_bytes = f.read()
        with open(template_path, "rb") as f:
            template_bytes = f.read()

        # Get subject info
        subject_name = self.reference_metadata.get("subject", person_name)
        features = self.reference_metadata.get("features", {})
        features_str = ", ".join([f"{k}: {v}" for k, v in features.items()]) if features else ""

        # Build the generation prompt
        full_prompt = f"""I have two images:
1. FIRST IMAGE: A meme template that I want to recreate
2. SECOND IMAGE: A reference photo of {subject_name}

YOUR TASK: Recreate the meme from the first image, but replace the face with {subject_name}'s face from the second image.

TEMPLATE ANALYSIS:
{template.prompt}

REQUIRED EXPRESSION: {template.expression_needed}
REQUIRED ANGLE: {template.angle_needed}

CRITICAL REQUIREMENTS FOR FACE PRESERVATION:
- The face must be EXACTLY {subject_name}'s face from the reference photo
- Preserve their exact: facial structure, skin tone, eye shape, nose shape, mouth, any facial hair
- KEY FEATURES TO PRESERVE: {features_str if features_str else "glasses, facial hair, distinctive features"}
- The face should be instantly recognizable as {subject_name}
- Match the lighting and angle to fit the meme composition naturally
- Adapt the expression to match what the meme requires while keeping the face recognizable

OUTPUT: A high-quality meme that looks like {subject_name} in this exact meme format."""

        try:
            response = self.client.models.generate_content(
                model=IMAGE_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=template_bytes, mime_type=self._get_mime_type(str(template_path))),
                            types.Part.from_bytes(data=person_bytes, mime_type=self._get_mime_type(str(reference_path))),
                            types.Part.from_text(text=full_prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                    temperature=1.0,
                ),
            )

            # Extract generated image
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    if output_path is None:
                        output_path = self.output_dir / f"{template.name.replace(' ', '_').lower()}.png"

                    with open(output_path, "wb") as f:
                        f.write(part.inline_data.data)

                    return output_path

            console.print(f"[yellow]No image generated for {template.name}[/yellow]")
            return None

        except Exception as e:
            console.print(f"[red]Error generating {template.name}: {e}[/red]")
            return None

    def generate_meme_pack(
        self,
        person_name: str = "person",
        templates: Optional[list[str]] = None,
        limit: Optional[int] = None,
        resume: bool = True,
    ) -> list[Path]:
        """Generate a full pack of memes for one person."""

        # Ensure all templates are analyzed
        self.analyze_all_templates()

        # Get templates to generate
        if templates:
            to_generate = [t for t in templates if t in self.templates]
        else:
            to_generate = list(self.templates.keys())

        if limit:
            to_generate = to_generate[:limit]

        # Create or find output directory for this pack
        subject = self.reference_metadata.get("subject", person_name).replace(" ", "_").lower()

        # Check for existing pack directory to resume
        pack_dir = None
        if resume:
            existing_packs = sorted(self.output_dir.glob(f"{subject}_memes_*"), reverse=True)
            if existing_packs:
                pack_dir = existing_packs[0]  # Use most recent
                console.print(f"[yellow]Resuming from existing pack: {pack_dir.name}[/yellow]")

        if pack_dir is None:
            pack_dir = self.output_dir / f"{subject}_memes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            pack_dir.mkdir(exist_ok=True)

        # Check which memes already exist (for resume functionality)
        already_done = []
        remaining = []
        for filename in to_generate:
            template = self.templates[filename]
            safe_name = template.name.replace(" ", "_").replace("/", "-").lower()
            output_path = pack_dir / f"{safe_name}.png"
            if output_path.exists():
                already_done.append((filename, output_path))
            else:
                remaining.append(filename)

        console.print(f"\n[bold cyan]Generating Meme Pack for: {self.reference_metadata.get('subject', person_name)}[/bold cyan]")
        console.print(f"[dim]Templates: {len(to_generate)} total | Already done: {len(already_done)} | Remaining: {len(remaining)}[/dim]")
        console.print(f"[dim]Output: {pack_dir}[/dim]\n")

        if already_done:
            console.print(f"[green]Skipping {len(already_done)} already-generated memes[/green]")

        generated = [path for _, path in already_done]  # Include already-done in results

        if not remaining:
            console.print(f"\n[bold green]All {len(to_generate)} memes already generated![/bold green]")
            console.print(f"[dim]Saved to: {pack_dir}[/dim]")
            return generated

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating memes...", total=len(remaining))

            for filename in remaining:
                template = self.templates[filename]
                progress.update(task, description=f"Creating: {template.name}")

                safe_name = template.name.replace(" ", "_").replace("/", "-").lower()
                output_path = pack_dir / f"{safe_name}.png"
                result = self.generate_meme(
                    filename,
                    person_name=person_name,
                    output_path=output_path,
                )

                if result:
                    generated.append(result)
                    console.print(f"  [green]✓[/green] {template.name}")
                else:
                    console.print(f"  [red]✗[/red] {template.name}")

                progress.advance(task)

        console.print(f"\n[bold green]Generated {len(generated)}/{len(to_generate)} memes![/bold green]")
        console.print(f"[dim]Saved to: {pack_dir}[/dim]")

        return generated

    def list_templates(self):
        """Display available meme templates."""
        self.analyze_all_templates()

        table = Table(title="Available Meme Templates")
        table.add_column("File", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Expression", style="yellow")
        table.add_column("Angle", style="magenta")
        table.add_column("Description", style="white")

        for filename, template in sorted(self.templates.items()):
            desc = template.description[:40] + "..." if len(template.description) > 40 else template.description
            table.add_row(
                filename,
                template.name,
                template.expression_needed,
                template.angle_needed,
                desc
            )

        console.print(table)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate personalized meme packs using Gemini Nano Banana Pro"
    )
    parser.add_argument(
        "--name",
        default="person",
        help="Name of the person (used in prompts and output folder)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available meme templates",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze all templates and cache prompts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of memes to generate",
    )

    args = parser.parse_args()

    generator = MemeGenerator()

    if args.list:
        generator.list_templates()
        return

    if args.analyze:
        generator.analyze_all_templates()
        console.print("\n[green]All templates analyzed and cached![/green]")
        return

    generator.generate_meme_pack(person_name=args.name, limit=args.limit)


if __name__ == "__main__":
    main()
