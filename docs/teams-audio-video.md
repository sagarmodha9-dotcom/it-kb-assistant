# Microsoft Teams Audio and Video Problems

Applies to: Teams desktop app. Owner: Collaboration team.

## Quick checks
1. In Teams, select Settings and more (...) > Settings > Devices, and choose the correct speaker, microphone, and camera.
2. Make a test call: Settings > Devices > Make a test call.
3. Close other apps that might be using the camera (for example Zoom or the Camera app).

## No audio in meetings
- Check that the headset is selected in Teams, not only in Windows or macOS.
- Unplug and reconnect USB headsets, then rejoin the meeting.

## Camera not working
- On Mac, open System Settings > Privacy & Security > Camera and allow Microsoft Teams.
- On Windows, open Settings > Privacy & security > Camera and allow desktop apps to access the camera.

## Clear the Teams cache
If problems continue, quit Teams fully, then clear the cache:
- Windows: delete the contents of %userprofile%\appdata\local\Packages\MSTeams_8wekyb3d8bbwe\LocalCache\Microsoft\MSTeams
- Mac: delete ~/Library/Group Containers/UBF8T346G9.com.microsoft.teams
Then reopen Teams and sign in again.
