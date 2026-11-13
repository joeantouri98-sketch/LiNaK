"""The logo drawn in code: icons, species colours, the Qali theme."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from qali.gui import brand, theme  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    store = QSettings(str(tmp_path / "s.ini"), QSettings.IniFormat)
    brand.load_colours(store)
    yield store
    brand._chosen.clear()


def _ember_pixel(icon, size=256):
    """The colour in the middle of the ember of *icon*."""
    image = icon.pixmap(size, size).toImage()
    return image.pixelColor(round(size * 0.48), round(size * 0.30)).name()


def test_alkalis_have_their_flame_colour_and_others_the_logo_orange():
    assert brand.species_colour("Na") == brand.FLAME_COLOURS["Na"]
    assert brand.species_colour("Na_c1") == brand.FLAME_COLOURS["Na"]      # an ion of Na
    assert brand.species_colour("Fe") == brand.EMBER
    assert brand.species_colour("") == brand.EMBER


def test_icon_ember_is_the_species_colour():
    assert _ember_pixel(brand.icon()) == brand.EMBER
    assert _ember_pixel(brand.species_icon("Cs")) == brand.FLAME_COLOURS["Cs"]
    assert set(brand.icon().availableSizes()[i].width() for i in range(len(brand.ICON_SIZES))) == set(brand.ICON_SIZES)


def test_a_chosen_colour_is_saved_and_default_clears_it(settings):
    brand.set_colour(settings, "Fe", "#3366ff")
    assert brand.species_colour("Fe") == "#3366ff"
    brand._chosen.clear()
    brand.load_colours(settings)                                           # as on the next start
    assert brand.species_colour("Fe") == "#3366ff"
    brand.set_colour(settings, "Fe", brand.default_colour("Fe"))
    assert brand.species_colour("Fe") == brand.EMBER
    assert not settings.contains(brand.COLOUR_SETTING.format(species="Fe"))


def test_wordmark_and_splash_paint_something():
    mark = brand.wordmark(24, "#000000").toImage()
    assert mark.height() == 48 and any(
        QColor(mark.pixel(x, mark.height() // 2)).alpha() for x in range(mark.width()))
    assert brand.splash("1.0", "tagline").toImage().pixelColor(4, 4).name() == brand.SPLASH_BG


def test_qali_theme_is_listed_and_dark():
    assert theme.QALI in theme.THEME_NAMES and theme.QALI in theme.COLOUR_THEMES
    assert theme.is_dark(theme.QALI) and theme.is_dark(theme.DARK) and not theme.is_dark(theme.LIGHT)
    assert theme.THEMES[theme.QALI].bg != theme.THEMES[theme.DARK].bg


@pytest.mark.parametrize("name", [theme.DARK, theme.QALI])
def test_dark_and_qali_use_the_ember_as_accent(name):
    assert theme.THEMES[name].accent == brand.EMBER


def test_light_keeps_its_original_green():
    assert theme.THEMES[theme.LIGHT].accent == "#8fe03f"
    assert theme.THEMES[theme.LIGHT].selection == "#e4f5d4"


@pytest.fixture
def qali_base():
    yield
    theme.set_qali_base(None)


def test_qali_theme_follows_its_base_colour(qali_base):
    assert theme.tokens_for(theme.QALI).bg == theme.QALI_BASE
    theme.set_qali_base("#2a2440")
    tokens = theme.tokens_for(theme.QALI)
    assert tokens.bg == "#2a2440" and tokens.surface != theme.THEMES[theme.QALI].surface
    assert theme.is_dark(theme.QALI)
    theme.set_qali_base("#e8ecef")                                        # a light base gives dark text
    assert not theme.is_dark(theme.QALI)
    assert QColor(theme.tokens_for(theme.QALI).text).lightness() < 80
    theme.set_qali_base("not-a-colour")
    assert theme.tokens_for(theme.QALI).bg == theme.QALI_BASE


def test_base_colour_is_saved_and_cancel_restores_it(qali_base, tmp_path):
    from qali.gui import preferences_dialog as P

    store = QSettings(str(tmp_path / "p.ini"), QSettings.IniFormat)
    dialog = P.PreferencesDialog(None, theme.QALI, store, lambda: None)
    dialog.base_choice = "#203040"
    dialog._ok()
    assert store.value(P.QALI_BASE_KEY) == "#203040"
    theme.set_qali_base(None)
    P.load_overrides(store)                                               # as on the next start
    assert theme.qali_base() == "#203040"
    dialog = P.PreferencesDialog(None, theme.QALI, store, lambda: None)
    dialog.base_choice = "#ffffff"
    dialog._apply()
    dialog._cancel()
    assert theme.qali_base() == "#203040"


def test_sprig_ember_takes_the_given_colour():
    image = brand.sprig(120, "#000000", "#8fe03f", ratio=1.0).toImage()
    assert image.pixelColor(60, 34).name() == "#8fe03f"                   # inside the ember
