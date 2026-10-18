# OneDrive Sync Problems

Applies to: OneDrive for Business. Owner: Collaboration team.

## Check the sync status
Click the OneDrive cloud icon in the taskbar (Windows) or menu bar (Mac).
- Blue cloud with arrows: syncing.
- Red X: a sync error. Click it to see which file has the problem.
- Paused: click "Resume syncing".

## Common causes
- File names with characters that are not allowed: " * : < > ? / \ |
- Files larger than 250 GB.
- File path longer than 400 characters (move the file to a shorter folder path).
- Storage full: each user has 1 TB. Check usage in OneDrive settings.

## Reset OneDrive
If sync is stuck, reset the app:
- Windows: press Windows + R and run %localappdata%\Microsoft\OneDrive\onedrive.exe /reset
- Mac: quit OneDrive, open Applications, right-click OneDrive, Show Package Contents > Contents > Resources, and run ResetOneDriveApp.command.
Resetting does not delete your files; OneDrive re-syncs everything, which can take a few hours.
