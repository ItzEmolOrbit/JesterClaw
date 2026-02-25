# --- Image processing (screenshots sent from client) ---
echo "[*] Installing image tools..."
pip install Pillow

# --- Utilities ---
echo "[*] Installing utilities..."
pip install psutil python-dotenv aiofiles httpx rich

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║    Dependencies installed successfully!      ║"
echo "║    Next: bash model_installer.bash           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
