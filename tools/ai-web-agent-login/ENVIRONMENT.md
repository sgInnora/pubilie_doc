# Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `TZ` | `Asia/Shanghai` | Timezone propagated to Firefox/Xvfb for consistent timestamps and logging. |
| `LANG` / `LC_ALL` | `zh_CN.UTF-8` | Ensures UTF-8 locale with Chinese font rendering inside the container. |
| `SCREEN_WIDTH` | `1920` | Virtual display width consumed by Firefox/VNC. Combine with height/depth for canvas stability. |
| `SCREEN_HEIGHT` | `1080` | Virtual display height. |
| `SCREEN_DEPTH` | `24` | Color depth for the Xvfb screen. |
| `ENABLE_VNC` | `true` | Keeps the VNC server enabled (port `5900`). Leave `true` for manual debugging. |
| `SE_VNC_NO_PASSWORD` | `1` | Removes the default VNC password so Synology can proxy the port without prompts. Set to `0` to enforce the Selenium default password. |
| `SESSION_TIMEOUT` | `900` | Idle timeout (seconds) for Selenium sessions. Handed to `SE_OPTS` by the entrypoint wrapper. |
| `STEALTH_USER_AGENT` | `Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0` | Spoofed user agent injected into the Firefox profile. Override when you need to mirror a specific browser build. |
| `STEALTH_PLATFORM` | `Win64` | Value reported by `navigator.platform`. Helps avoid automation fingerprints. |
| `ANTI_DETECTION_PROFILE` | `true` | Enables profile hardening (disables `navigator.webdriver`, caching, mic/cam hints, etc.). Set to `false` to obtain a vanilla profile. |
| `FIREFOX_PROFILE_DIR` | `/home/seluser/.mozilla/firefox/automation.default` | Profile path that gets populated/persisted. Update if you mount a different directory. |
| `MOZ_HEADLESS` | `0` | Forces Firefox to draw to the virtual display (`0`). Set to `1` if you only need the Selenium HTTP API without VNC/noVNC. |
| `SE_OPTS` | _empty_ | Extra options passed straight to Selenium. The entrypoint always prefixes `--session-timeout <value>` and then appends this variable. |

All variables above can be overridden per environment in `docker-compose.yml` or via `export VAR=value` before running `./start-stack.sh`.
