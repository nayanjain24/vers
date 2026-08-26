#

## 🚨 Stability & Recovery (v5.0 Deep Fix)

This version of VERS has been hardened for macOS stability.

- **Direct Mode**: If the hardware camera is blocked, the dashboard automatically provides a **Direct Browser Mode**. Capture a frame in the main slot to run the full AI pipeline.
- **Fix Camera**: Run `bash fix_camera.sh` to reset macOS permissions.
- **Full Restore**: If the virtual environment breaks, run `bash restore_env.sh` to rebuild the exact working dependency matrix.
