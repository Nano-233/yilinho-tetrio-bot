# Python Tetris bot for TETR.IO
This bot retrieves game data through color matching and calculates optimal moves using multiprocessing. It's optimized for back-to-back (b2b) moves, including T-spins and other advanced spins.

## Demo
https://youtu.be/nqyY3mnWVAE

## Quick Start

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the calibration wizard (first time setup):
    ```bash
    python bot.py --calibrate
    ```

3. Start the bot with your saved configuration:
    ```bash
    python bot.py --use-config
    ```

## Calibration

The bot includes an interactive calibration wizard that captures your screen coordinates for accurate gameplay detection. This is much easier than manually editing coordinate values!

### macOS note

Global hotkeys (`=` / Escape) from the `keyboard` library do **not** work on macOS without root, and often not even then. This fork uses a **countdown capture** instead: press Enter in the terminal, swipe to TETR.IO, hover the spot, wait for capture. Key presses use `pyautogui` (needs Accessibility). Screen grabs use `mss` (needs Screen Recording).

### Running Calibration

```bash
python3 bot.py --calibrate
```

### Calibration Steps

The wizard guides you through 5 steps to capture the necessary screen positions:

| Step | What to Capture | Description |
|------|-----------------|-------------|
| 1 | Board Top-Left | Move mouse to the top-left corner of the Tetris board |
| 2 | Board Bottom-Right | Move mouse to the bottom-right corner of the Tetris board |
| 3 | Next Piece #1 | Move mouse to the center of the first (topmost) next piece preview |
| 4 | Next Piece #5 | Move mouse to the center of the fifth (bottommost) next piece preview |
| 5 | Held Piece | Move mouse to the center of the held piece display |

### Controls During Calibration

- **Press Enter** in the terminal to start a 5-second countdown
- During the countdown, switch to TETR.IO and hover the target
- Keep the mouse still until it prints `Captured: (x, y)`

### Configuration File

After successful calibration, your settings are saved to `config.json`. This file stores:
- Screen resolution
- Board coordinates
- Next piece positions
- Held piece position
- AI parameters (multiprocessing workers, pruning settings)

## Usage

### Command Line Options

| Option | Description |
|--------|-------------|
| `--calibrate` | Run the interactive calibration wizard |
| `--use-config` | Load settings from `config.json` |
| `--mp N` | Override multiprocessing workers (default: 4) |
| `--pruning-moves N` | Override pruning moves parameter |
| `--pruning-breadth N` | Override pruning breadth parameter |
| `--delay N` | Override move delay in milliseconds (default: 30) |
| `--action-delay N` | Override action delay in milliseconds (default: 50) |
| `--delay-variance N` | Override delay variance percentage (default: 20) |

### Examples

```bash
# First time setup - calibrate your screen
python bot.py --calibrate

# Run bot with saved configuration
python bot.py --use-config

# Run with custom performance settings
python bot.py --use-config --mp 8

# Run with custom AI parameters
python bot.py --use-config --mp 8 --pruning-moves 3 --pruning-breadth 5

# Run with custom delay settings (more human-like)
python bot.py --use-config --delay 50 --action-delay 80 --delay-variance 30

# Run without config (uses hardcoded defaults)
python bot.py
```

## Delay Settings (Anti-Cheat)

The bot includes configurable delays to make inputs appear more human-like, which helps avoid anti-cheat detection.

### Delay Parameters

| Setting | Description | Default |
|---------|-------------|---------|
| `move_delay_ms` | Delay between each keypress (left/right movements) | 30ms |
| `action_delay_ms` | Delay after actions like hold, rotate, and hard drop | 50ms |
| `delay_variance_percent` | Random variance applied to delays (±%) | 20% |

### How Variance Works

The variance makes timing less predictable. For example, with a 30ms base delay and 20% variance:
- Actual delays will range from 24ms to 36ms (30ms ± 20%)
- Each delay is randomly calculated within this range

### Configuration Methods

1. **During Calibration**: The wizard prompts for delay settings in step 6
2. **In config.json**: Manually edit the delay values
3. **CLI Override**: Use `--delay`, `--action-delay`, or `--delay-variance` flags

### Example config.json with delays

```json
{
  "screen_offset": [0, 0],
  "screen_resolution": [1920, 1080],
  "board_top_left": [730, 82],
  "board_bottom_right": [1185, 988],
  "next_piece_xy_0": [1337, 192],
  "next_piece_xy_4": [1337, 735],
  "held_piece_xy": [615, 191],
  "move_delay_ms": 30,
  "action_delay_ms": 50,
  "delay_variance_percent": 20
}
```

### Recommended Delay Values

| Use Case | move_delay_ms | action_delay_ms | variance |
|----------|---------------|-----------------|----------|
| Maximum Speed (risky) | 10 | 20 | 10 |
| Balanced (default) | 30 | 50 | 20 |
| Safe/Human-like | 50 | 80 | 30 |
| Very Conservative | 80 | 120 | 40 |

### Manual Configuration (Legacy)

If you prefer not to use the calibration wizard, you can still manually adjust the parameters in `bot.py`:
```python
screen_resolution=(1920, 1080),
board_top_left=(787, 220),
board_bottom_right=(1133, 899),
next_piece_xy_0=(1260, 300),
next_piece_xy_4=(1260, 721),
held_piece_xy=(691, 300),
```

## Game Settings

For optimal bot performance, configure TETR.IO with these settings:
- **ARR**: 0ms
- **DAS**: 40ms
- **SDF**: max

## Keybinds

The bot sends real keystrokes, so its binds **must match your TETR.IO controls**.
Defaults are TETR.IO's stock layout:

| Action | Default |
|--------|---------|
| Move left / right | `left` / `right` |
| Soft drop | `down` |
| Hard drop | `space` |
| Rotate CW | `x` |
| Rotate CCW | `z` |
| Rotate 180 | `a` |
| Hold | `c` |

The calibration wizard prompts for these. You can also edit `keybinds` in `config.json`:

```json
"keybinds": {
  "move_left": "a",
  "move_right": "d",
  "soft_drop": "s",
  "hard_drop": "space",
  "rotate_cw": "right",
  "rotate_ccw": "left",
  "rotate_180": "up",
  "hold": "shift"
}
```

## Verifying detection

Before letting the bot play, check that it reads the board correctly:

```bash
python3 bot.py --use-config --test
```

It prints the detected next queue, held piece, and filled-cell count once a second
without pressing any keys. If `held=None` persists or the queue looks wrong,
re-run `--calibrate` and aim at the **colored blocks**, not the preview panel.

## Troubleshooting

### Calibration Issues

- **Mouse position not capturing**: Ensure no other application is intercepting the `=` key
- **Wrong coordinates saved**: Re-run `python bot.py --calibrate` to recalibrate
- **Config file not loading**: Check that `config.json` exists and is valid JSON

### Bot Performance

- **Bot moves incorrectly**: Re-calibrate to ensure accurate board detection
- **Bot is slow**: Increase `--mp` value for more parallel processing
- **Bot misses pieces**: Ensure the next piece and held piece positions are correctly calibrated
- **`Held is None` then it freezes**: fixed — the held piece is identified by hue
  now, so TETR.IO greying it out after a hold no longer breaks detection. The
  loop also re-syncs the next queue instead of waiting forever for a change
  that a locked hold key can never produce.

## Dependencies

- Python 3.x
- pyautogui
- mss
- numpy
- See `requirements.txt` for full list

## Disclaimer

Use this bot at your own discretion. Using it in multiplayer mode could result in your account and IP address being banned.
