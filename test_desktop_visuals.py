"""Opt-in fake-session UI checks and a model-free visual preview."""
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import desktop_app as desktop
import app_settings as settings
import launch_gemma as launch
from ui_theme import install_theme


class FakeWorker:
    def __init__(self, args, preferences):
        self.output = queue.Queue()
        self.closed = threading.Event()
        self.commands = []

    def start(self):
        pass

    def send(self, action, **fields):
        self.commands.append((action, fields))
        if action == 'quit':
            self.closed.set()


def destroy_root(root):
    for timer in root.tk.splitlist(root.tk.call('after', 'info')):
        root.after_cancel(timer)
    root.destroy()


def snapshot(app, state, *, phase='Running', server='Ready', running=True):
    app.handle_event(dict(event='snapshot', state=state, phase=phase, server=server,
                          running=running, detail='', checks={'Microphone': 'Ready'},
                          logs=['Local services ready'], resources='RAM: 12 GB free'))


def make_app(root, directory):
    path = Path(directory) / 'settings.json'
    settings.save_preferences(dict(settings.DEFAULTS, device='Preview microphone'), path)
    app = desktop.DesktopApp(root, launch.parse_args(['--wait', '--settings-file', str(path)]),
                            worker_factory=FakeWorker, discover=lambda: (['Preview microphone'], ''))
    deadline = time.monotonic() + 2
    while app.refreshing and time.monotonic() < deadline:
        root.update()
        time.sleep(.01)
    return app


@unittest.skipUnless(os.environ.get('GEMMA_DESKTOP_TESTS') == '1', 'opens a simulated desktop session')
class VisualWindowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.root = desktop.tk.Tk()
        self.addCleanup(destroy_root, self.root)
        self.app = make_app(self.root, self.directory.name)

    def test_session_states_actions_and_shared_asset_cache(self):
        app = self.app
        self.assertFalse(app.settings_open)
        self.assertTrue(app.pause_button.instate(['disabled']))
        self.assertTrue(app.cancel_button.instate(['disabled']))
        original_assets = tuple(app.theme_images)
        snapshot(app, 'Stopped', phase='Loading', running=False)
        self.assertTrue(app.start_button.instate(['disabled']))
        for state in ('Listening', 'Thinking', 'Preparing voice', 'Speaking', 'Paused'):
            snapshot(app, state)
            self.root.update()
            self.assertTrue(app.pause_button.winfo_ismapped())
            self.assertFalse(app.start_button.winfo_ismapped())
            self.assertEqual(app.cancel_button.instate(['!disabled']),
                             state in ('Thinking', 'Preparing voice', 'Speaking'))
            self.assertEqual(app.pause_button['text'], 'Resume listening' if state == 'Paused' else 'Pause listening')
        app.pause_button.invoke()
        self.assertEqual(app.worker.commands[-1][0], 'resume')
        snapshot(app, 'Speaking')
        app.cancel_button.invoke()
        self.assertEqual(app.worker.commands[-1][0], 'cancel')
        snapshot(app, 'Error', phase='Error', running=False)
        self.assertEqual(app.start_button['text'], 'Retry')
        snapshot(app, 'Error', server='Unavailable')
        self.assertEqual(app.start_button['text'], 'Reconnect')
        app.start_button.invoke()
        self.assertEqual(app.worker.commands[-1][0], 'restart')
        self.assertEqual(tuple(app.theme_images), original_assets)
        self.assertIs(install_theme(self.root)[1], app.theme_images)

    def test_settings_sharing_copy_clear_and_scroll_preservation(self):
        app = self.app
        app.settings_button.invoke()
        self.root.update()
        self.assertTrue(app.settings_open)
        self.assertFalse(app.chat_frame.winfo_ismapped())
        for title in app.sections:
            app.toggle_section(title)
            self.root.update()
            self.assertTrue(app.sections[title][0].winfo_ismapped())
            app.toggle_section(title)
        app.voice.set(False)
        app.apply_button.invoke()
        self.assertFalse(app.worker.commands[-1][1]['preferences']['voice'])
        app.back_button.invoke()
        self.root.update()
        self.assertTrue(app.chat_frame.winfo_ismapped())
        app.share_button.invoke()
        self.assertEqual(app.sharing_text.get(), 'Share screen · On')
        self.assertEqual(app.worker.commands[-1], ('screen_sharing', {'enabled': True}))
        app.share_button.invoke()
        self.assertEqual(app.sharing_text.get(), 'Share screen · Off')
        self.assertEqual(app.worker.commands[-1], ('screen_sharing', {'enabled': False}))
        for number in range(20):
            app.handle_event(dict(event='turn_started', turn_id=str(number)))
            app.handle_event(dict(event='turn_finished', turn_id=str(number), text='A readable answer. ' * 40, complete=True))
        app.render_conversation()
        self.root.update()
        app.copy_buttons[0].invoke()
        self.assertEqual(app.copy_buttons[0]['text'], 'Copied')
        self.assertTrue(self.root.clipboard_get().startswith('A readable answer.'))
        app.chat.yview_moveto(.2)
        self.root.update()
        before = app.chat.index('@0,0')
        app.handle_event(dict(event='turn_started', turn_id='last'))
        app.render_conversation()
        self.root.update()
        self.assertEqual(app.chat.index('@0,0'), before)
        app.clear_button.invoke()
        self.assertEqual(app.worker.commands[-1][0], 'clear')
        app.handle_event(dict(event='cleared'))
        app.render_conversation()
        self.assertIn('Speak when', app.chat.get('1.0', 'end'))
        # A pending copied-feedback callback must tolerate a destroyed button.
        self.root.after(1550, self.root.quit)
        self.root.mainloop()

    def test_minimum_window_at_multiple_scalings(self):
        for factor in (1, 1.5, 2):
            with self.subTest(scale=factor):
                root = desktop.tk.Tk()
                try:
                    root.tk.call('tk', 'scaling', factor * 96 / 72)
                    app = make_app(root, self.directory.name)
                    width, height = root.minsize()
                    root.geometry(f'{width}x{height}')
                    snapshot(app, 'Listening')
                    root.update()
                    self.assertGreater(app.chat.winfo_height(), 80)
                    app.handle_event(dict(event='turn_started', turn_id='scaling', input_kind='text', input_text='Hello'))
                    app.handle_event(dict(event='turn_finished', turn_id='scaling', complete=True,
                                          text='A readable answer that wraps naturally. ' * 5))
                    app.render_conversation()
                    app.jump_latest()
                    root.update()
                    button = app.copy_buttons[0]
                    self.assertGreaterEqual(button.winfo_y(), 0)
                    self.assertLessEqual(button.winfo_y() + button.winfo_height(), app.chat.winfo_height())
                    reply_line = app.chat.dlineinfo(app.rendered_turns['scaling']['reply_end'] + '-2c')
                    self.assertIsNotNone(reply_line)
                    self.assertGreaterEqual(reply_line[1], 0)
                    for widget in (app.settings_button, app.quit_button, app.pause_button,
                                   app.cancel_button, app.share_button, app.clear_button):
                        self.assertGreaterEqual(widget.winfo_rootx(), root.winfo_rootx())
                        self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                             root.winfo_rootx() + root.winfo_width())
                    app.toggle_settings()
                    app.toggle_section('Server configuration')
                    root.update()
                    self.assertGreater(app.advanced_canvas.winfo_height(), 80)
                    self.assertLessEqual(app.server_section.winfo_reqwidth(), app.advanced_canvas.winfo_width())
                    self.assertTrue(app.apply_button.winfo_ismapped())
                    self.assertLessEqual(app.apply_button.winfo_rooty() + app.apply_button.winfo_height(),
                                         root.winfo_rooty() + root.winfo_height())
                finally:
                    destroy_root(root)


def preview(factor, show_settings=False):
    # Preview never uses services, capture, microphone discovery, or real preferences.
    root = desktop.tk.Tk()
    root.tk.call('tk', 'scaling', factor * 96 / 72)
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        with patch.object(desktop.ctypes.windll.user32, 'SetWindowDisplayAffinity'):
            app = make_app(root, directory)
        root.title(f'Gemma Live — Visual preview {factor:g}x')
        root.attributes('-topmost', True)
        width, height = root.minsize()
        root.geometry(f'{width}x{height}+40+40')
        snapshot(app, 'Listening')
        app.message.set('Ready. Speak naturally; Gemma pauses the microphone while replying.')
        app.handle_event(dict(event='turn_started', turn_id='preview', screen_sharing=False,
                              input_kind='text', input_text='What can you help me with?'))
        app.handle_event(dict(event='turn_finished', turn_id='preview', complete=True,
            text='Your assistant is ready. Ask a question, talk through an idea, or share your screen when you want help with what you see.'))
        app.render_conversation()
        snapshot(app, 'Listening')
        root.after_idle(app.jump_latest)
        if show_settings:
            app.toggle_settings()
            app.toggle_section('Speech detection')
        root.after(90000, app.close)
        root.mainloop()


if __name__ == '__main__':
    import sys
    if '--preview' in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(description='Preview the UI without local services')
        parser.add_argument('--preview', type=float, default=1)
        parser.add_argument('--settings', action='store_true')
        options = parser.parse_args()
        preview(options.preview, options.settings)
    else:
        unittest.main()
