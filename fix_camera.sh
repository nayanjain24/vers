#!/bin/bash
echo "Attempting to reset macOS Camera permissions for Terminal/IDE..."
tccutil reset Camera
echo "Permissions reset. Please restart the app and click 'Allow' when prompted."
