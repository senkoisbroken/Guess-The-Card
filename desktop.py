"""Windows foreground, client bounds, and owned input only. No process handle."""
import ctypes
import time
from ctypes import wintypes as w

ULONG_PTR = ctypes.c_size_t


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [('dx',w.LONG),('dy',w.LONG),('mouseData',w.DWORD),
               ('dwFlags',w.DWORD),('time',w.DWORD),('dwExtraInfo',ULONG_PTR)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [('wVk',w.WORD),('wScan',w.WORD),('dwFlags',w.DWORD),
               ('time',w.DWORD),('dwExtraInfo',ULONG_PTR)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [('uMsg',w.DWORD),('wParamL',w.WORD),('wParamH',w.WORD)]


class INPUT_UNION(ctypes.Union):
    _fields_ = [('mi',MOUSEINPUT),('ki',KEYBDINPUT),('hi',HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ('data',)
    _fields_ = [('type',w.DWORD),('data',INPUT_UNION)]


class WindowsDesktop:
    def __init__(self):
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        u = self.user32
        u.GetForegroundWindow.restype = w.HWND
        u.GetClassNameW.argtypes = [w.HWND,w.LPWSTR,ctypes.c_int]
        u.GetWindowTextW.argtypes = [w.HWND,w.LPWSTR,ctypes.c_int]
        u.GetClientRect.argtypes = [w.HWND,ctypes.POINTER(w.RECT)]
        u.ClientToScreen.argtypes = [w.HWND,ctypes.POINTER(w.POINT)]
        u.IsIconic.argtypes = [w.HWND]
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.GetAsyncKeyState.restype = w.SHORT
        u.GetCursorPos.argtypes = [ctypes.POINTER(w.POINT)]
        u.GetCursorPos.restype = w.BOOL
        u.SendInput.argtypes = [w.UINT,ctypes.POINTER(INPUT),ctypes.c_int]
        u.SendInput.restype = w.UINT

    def prepare_dpi(self):
        try:
            self.user32.SetProcessDpiAwarenessContext.argtypes = [w.HANDLE]
            self.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except (AttributeError,OSError):
            self.user32.SetProcessDPIAware()

    def roblox_window(self):
        u = self.user32
        hwnd = u.GetForegroundWindow()
        if not hwnd or u.IsIconic(hwnd):
            return None
        name, title = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(512)
        if not u.GetClassNameW(hwnd, name, len(name)) or not u.GetWindowTextW(hwnd, title, len(title)):
            return None
        return hwnd if name.value == 'WINDOWSCLIENT' and title.value.lower().startswith('roblox') else None

    def client_rect(self, hwnd):
        rect, point = w.RECT(), w.POINT()
        if not self.user32.GetClientRect(hwnd,ctypes.byref(rect)) or not self.user32.ClientToScreen(hwnd,ctypes.byref(point)):
            raise OSError('Client bounds unavailable')
        if rect.right < 64 or rect.bottom < 64:
            raise ValueError('Game window is too small')
        return point.x, point.y, rect.right, rect.bottom

    def down(self, key):
        return bool(self.user32.GetAsyncKeyState(key) & 0x8000)

    def _send(self, packet):
        if self.user32.SendInput(1,ctypes.byref(packet),ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def key_button(self, vk, down):
        scans = {32:0x39,67:0x2e,86:0x2f,66:0x30,69:0x12,87:0x11,65:0x1e,83:0x1f,68:0x20}
        if vk not in scans:
            raise ValueError('Unsupported output key')
        packet = INPUT(type=1)
        packet.ki.wScan = scans[vk]
        packet.ki.dwFlags = 8 | (0 if down else 2)
        self._send(packet)

    def mouse_button(self, down, right=False):
        packet = INPUT(type=0)
        packet.mi.dwFlags = (8 if down else 16) if right else (2 if down else 4)
        self._send(packet)

    def press_key(self, hwnd, bounds, vk, allowed):
        """Send an owned key press only to the current foreground game."""
        if not allowed() or self.roblox_window() != hwnd or self.client_rect(hwnd) != bounds or self.down(vk):
            return False
        self.key_button(vk, True)
        try:
            time.sleep(.06)
        finally:
            self.key_button(vk, False)
        return True

    def click_client(self, hwnd, bounds, x, y, allowed, before_press=None):
        """Click a detected client point only while the same game stays foreground."""
        left, top, width, height = bounds
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError('Point is outside the game window')
        if not allowed() or self.roblox_window() != hwnd or self.client_rect(hwnd) != bounds:
            return False
        # Small relative movements were verified on the live room map. They
        # keep Roblox's cursor tracking in step with the Windows cursor.
        target_x, target_y = left + x, top + y
        deadline = time.monotonic() + 25.
        while time.monotonic() < deadline:
            if not allowed() or self.roblox_window() != hwnd or self.client_rect(hwnd) != bounds:
                return False
            if self.down(1):
                return False
            point = w.POINT()
            if not self.user32.GetCursorPos(ctypes.byref(point)):
                raise ctypes.WinError(ctypes.get_last_error())
            dx, dy = target_x - point.x, target_y - point.y
            if abs(dx) <= 1 and abs(dy) <= 1:
                break
            movement = INPUT(type=0)
            movement.mi.dx = max(-8, min(8, dx))
            movement.mi.dy = max(-8, min(8, dy))
            movement.mi.dwFlags = 0x0001 | 0x2000
            self._send(movement)
            time.sleep(.001)
        else:
            return False
        time.sleep(.03)
        if not allowed() or self.roblox_window() != hwnd or self.client_rect(hwnd) != bounds or self.down(1):
            return False
        if before_press is not None and not before_press():
            return False
        if not allowed() or self.roblox_window() != hwnd or self.client_rect(hwnd) != bounds or self.down(1):
            return False
        self.mouse_button(True)
        try:
            time.sleep(.06)
        finally:
            self.mouse_button(False)
        return True
