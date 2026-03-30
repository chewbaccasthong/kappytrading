"""One-time setup script: generates RSA key pair and creates .env file.

Run this first before starting the bot:
    python setup.py

Then:
1. Copy the PUBLIC key into your Kalshi account API settings
2. Set your API Key ID in the .env file
3. Run: python main.py
"""

from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from rich.console import Console
from rich.panel import Panel

console = Console()


def generate_key_pair():
    """Generate RSA-2048 key pair for Kalshi API signing."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Save private key (keep this secret, never share)
    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Get public key to upload to Kalshi
    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return priv_pem, pub_pem


def main():
    console.print(Panel.fit(
        "[bold cyan]KAPPY TRADING BOT — SETUP[/bold cyan]",
        border_style="cyan",
    ))

    # Generate keys
    priv_key_path = Path("kalshi_private_key.pem")
    pub_key_path = Path("kalshi_public_key.pem")

    if priv_key_path.exists():
        console.print(f"[yellow]Private key already exists at {priv_key_path} — skipping keygen[/yellow]")
        priv_pem = priv_key_path.read_bytes()
        pub_pem = pub_key_path.read_bytes() if pub_key_path.exists() else b""
    else:
        console.print("[bold]Generating RSA-2048 key pair...[/bold]")
        priv_pem, pub_pem = generate_key_pair()
        priv_key_path.write_bytes(priv_pem)
        pub_key_path.write_bytes(pub_pem)
        priv_key_path.chmod(0o600)  # restrict permissions
        console.print(f"[green]✓ Private key saved to {priv_key_path}[/green]")
        console.print(f"[green]✓ Public key saved to {pub_key_path}[/green]")

    # Create .env if it doesn't exist
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(Path(".env.example").read_text())
        console.print(f"[green]✓ Created .env from .env.example[/green]")
    else:
        console.print(f"[yellow].env already exists — not overwriting[/yellow]")

    # Print instructions
    console.print()
    console.print(Panel(
        "[bold]Next steps:[/bold]\n\n"
        "1. [cyan]Create a Kalshi Demo account:[/cyan]\n"
        "   https://demo.kalshi.co/sign-up\n\n"
        "2. [cyan]Go to API settings:[/cyan]\n"
        "   https://demo.kalshi.co/account/api\n"
        "   → Click 'Add Key'\n"
        "   → Paste the contents of [bold]kalshi_public_key.pem[/bold]\n"
        "   → Copy the API Key ID shown\n\n"
        "3. [cyan]Edit your .env file:[/cyan]\n"
        "   Set [bold]KALSHI_API_KEY[/bold] = your API Key ID from step 2\n\n"
        "4. [cyan]Test the connection:[/cyan]\n"
        "   python main.py --once\n\n"
        "5. [cyan]Start the bot (dry run):[/cyan]\n"
        "   python main.py\n\n"
        "6. [cyan]When ready for live trading:[/cyan]\n"
        "   python main.py --live",
        title="Instructions",
        border_style="green",
    ))

    console.print()
    console.print("[bold yellow]Your public key (copy this into Kalshi API settings):[/bold yellow]")
    console.print(pub_pem.decode() if pub_pem else pub_key_path.read_text())


if __name__ == "__main__":
    main()
