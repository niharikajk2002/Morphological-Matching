# Base image for cross-compilation with Wine
FROM cdrx/pyinstaller-windows:python3

# Set working directory
WORKDIR /app

# Copy all necessary files
COPY . .

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || true \
    && pip check

# Use PyInstaller to create the Windows EXE
RUN pyinstaller --onefile --name finalgui --distpath /dist finalgui.py

# Default command
CMD ["bash"]
