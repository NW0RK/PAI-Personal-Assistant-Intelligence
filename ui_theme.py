"""Shared static Tk styling. Assets are generated once per Tcl interpreter."""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageTk


COLORS = dict(background='#11181D', panel='#1A252C', border='#34454F',
              text='#EDF3F5', muted='#A9BAC3', accent='#5DD4BE',
              warning='#F4C66A', error='#FF9696')
SPACE = dict(page=20, section=16, gap=8)


def install_theme(root):
    if hasattr(root, '_gemma_theme'):
        return root._gemma_theme
    c = COLORS
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Segoe UI', 10), background=c['background'], foreground=c['text'])
    for name, bg in [('TFrame', c['background']), ('Panel.TFrame', c['panel'])]:
        style.configure(name, background=bg)
    style.configure('TLabel', background=c['background'], foreground=c['text'])
    style.configure('Muted.TLabel', foreground=c['muted'])
    style.configure('Title.TLabel', font=('Segoe UI', 23, 'bold'))
    style.configure('Section.TLabel', font=('Segoe UI', 12, 'bold'))
    style.configure('State.TLabel', font=('Segoe UI', 15, 'bold'), foreground=c['accent'])
    style.configure('Panel.TLabel', background=c['panel'])
    style.configure('PanelMuted.TLabel', background=c['panel'], foreground=c['muted'])
    style.configure('TCheckbutton', background=c['background'], foreground=c['text'], padding=(0, 5))
    style.map('TCheckbutton', background=[('active', c['background'])],
              indicatorbackground=[('selected', c['accent']), ('!selected', c['border'])],
              foreground=[('disabled', c['muted'])])
    for name in ('TEntry', 'TCombobox'):
        style.configure(name, fieldbackground=c['panel'], foreground=c['text'],
                        bordercolor=c['border'], insertcolor=c['text'], padding=6,
                        lightcolor=c['border'], darkcolor=c['border'], arrowcolor=c['muted'])
        style.map(name, fieldbackground=[('readonly', c['panel'])],
                  foreground=[('readonly', c['text'])], bordercolor=[('focus', c['accent'])])
    root.option_add('*TCombobox*Listbox.background', c['panel'])
    root.option_add('*TCombobox*Listbox.foreground', c['text'])
    root.option_add('*TCombobox*Listbox.selectBackground', c['border'])
    root.option_add('*TCombobox*Listbox.selectForeground', c['text'])
    style.configure('TLabelframe', background=c['background'], bordercolor=c['border'],
                    lightcolor=c['border'], darkcolor=c['border'], relief='solid', borderwidth=1)
    style.configure('TLabelframe.Label', background=c['background'], foreground=c['muted'])
    style.configure('Horizontal.TProgressbar', background=c['accent'], troughcolor=c['border'],
                    borderwidth=0, bordercolor=c['border'], troughrelief='flat',
                    lightcolor=c['accent'], darkcolor=c['accent'])
    style.configure('Vertical.TScrollbar', background=c['border'], troughcolor=c['panel'],
                    borderwidth=0, arrowcolor=c['muted'], bordercolor=c['panel'],
                    lightcolor=c['border'], darkcolor=c['border'], gripcount=0)
    style.map('Vertical.TScrollbar', background=[('active', '#4B6571'), ('pressed', '#4B6571')])
    images = []
    scale = float(root.tk.call('tk', 'scaling')) / (96 / 72)
    edge = max(6, round(7 * scale))
    size = edge * 4

    def asset(fill, outline):
        # Supersample a small nine-slice sprite; ttk stretches its center.
        bitmap = Image.new('RGBA', (size * 3, size * 3))
        ImageDraw.Draw(bitmap).rounded_rectangle((1, 1, size * 3 - 2, size * 3 - 2),
            radius=edge * 3, fill=fill, outline=outline, width=3)
        photo = ImageTk.PhotoImage(bitmap.resize((size, size), Image.Resampling.LANCZOS), master=root)
        images.append(photo)
        return photo

    for name, fill, hover, pressed, fg in (
        ('TButton', '#24343D', '#304750', '#1E3038', c['text']),
        ('Primary.TButton', c['accent'], '#81E1CF', '#45B6A1', c['background']),
        ('Quiet.TButton', c['panel'], '#283A43', '#213039', c['text']),
        ('Copy.TButton', c['panel'], '#283A43', '#213039', c['muted'])):
        normal = asset(fill, c['border'] if name != 'Primary.TButton' else fill)
        disabled = asset(c['panel'], c['border'])
        focus = asset(fill, c['accent'] if name != 'Primary.TButton' else c['text'])
        active = asset(hover, c['accent'])
        down = asset(pressed, c['accent'])
        element = name.replace('.TButton', '').replace('TButton', 'Default') + '.rounded'
        style.element_create(element, 'image', normal, ('disabled', disabled), ('pressed', down),
                             ('focus', focus), ('active', active), border=edge, sticky='nsew')
        style.layout(name, [(element, {'sticky': 'nsew', 'children': [
            ('Button.padding', {'sticky': 'nsew', 'children': [('Button.label', {'sticky': 'nsew'})]})]})])
        style.configure(name, foreground=fg, background=c['panel'], borderwidth=0, width=0,
                        padding=(10, 4) if name != 'Copy.TButton' else (6, 1))
        style.map(name, foreground=[('disabled', '#879AA4')],
                  background=[('disabled', c['panel']), ('active', c['panel'])])
    def check_asset(selected=False, focused=False, disabled=False):
        size = max(16, round(18 * scale))
        gap = max(6, round(6 * scale))
        bitmap = Image.new('RGBA', ((size + gap) * 3, size * 3))
        draw = ImageDraw.Draw(bitmap)
        fill = c['accent'] if selected and not disabled else c['panel']
        outline = c['accent'] if focused else c['border']
        draw.rounded_rectangle((2, 2, size * 3 - 3, size * 3 - 3), radius=9,
                               fill=fill, outline=outline, width=3)
        if selected:
            draw.line([(size * .7, size * 1.5), (size * 1.3, size * 2.1),
                       (size * 2.3, size * .9)], fill=c['muted'] if disabled else c['background'],
                      width=max(3, round(size * .22)))
        photo = ImageTk.PhotoImage(bitmap.resize((size + gap, size), Image.Resampling.LANCZOS), master=root)
        images.append(photo)
        return photo

    check = check_asset()
    style.element_create('Gemma.check', 'image', check,
                         ('disabled', 'selected', check_asset(selected=True, disabled=True)),
                         ('selected', check_asset(selected=True)), ('focus', check_asset(focused=True)))
    style.layout('TCheckbutton', [('Checkbutton.padding', {'sticky': 'nsew', 'children': [
        ('Gemma.check', {'side': 'left', 'sticky': ''}),
        ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [
            ('Checkbutton.label', {'sticky': 'nsew'})]})]})])
    style.configure('TCheckbutton', padding=(0, 5), indicatorspace=6)
    root.configure(bg=c['background'])
    root._gemma_theme = (style, images)
    return root._gemma_theme
