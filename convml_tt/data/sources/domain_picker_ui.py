"""
Simple UI to use in jupyter notebook to draw a domain on the global world map
for use in tile generation

In a jupyter notebook cells simply run the command:

>>> DomainPicker().m
"""

try:
    from ipyleaflet import (
        Map,
        Marker,
        Polygon,
        WidgetControl,
        basemap_to_tiles,
        basemaps,
    )
except ImportError:
    raise Exception(
        "To use the domain picker UI you will need to install" " `ipyleaflet`"
    )

import datetime
from functools import partial

import yaml
from ipywidgets import Button, Text


class DomainPicker:
    def __init__(self, n_days_old_satimg=1):
        t = datetime.datetime.now() - datetime.timedelta(days=n_days_old_satimg)
        t_str = t.strftime("%Y-%m-%d")

        self.m = Map(
            layers=[
                basemap_to_tiles(basemaps.NASAGIBS.ModisTerraTrueColorCR, t_str),
            ],
            center=(52.204793, 360.121558),
            zoom=2,
        )

        self.domain_coords = []
        self.polygon = None
        self.marker_locs = {}

        self.m.on_interaction(self._handle_map_click)

        button_reset = Button(description="reset")
        button_reset.on_click(self._clear_domain)
        button_save = Button(description="save domain")
        button_save.on_click(self._save_domain)
        self.name_textfield = Text(value="domain_name", width=10)

        self.m.add_control(WidgetControl(widget=button_save, position="bottomright"))
        self.m.add_control(WidgetControl(widget=button_reset, position="bottomright"))
        self.m.add_control(
            WidgetControl(widget=self.name_textfield, position="bottomright")
        )

    def _update_domain_render(self):
        if self.polygon is not None:
            self.m.remove_layer(self.polygon)

        if len(self.domain_coords) > 1:
            self.polygon = Polygon(
                locations=self.domain_coords, color="green", fill_color="green"
            )
            self.m.add_layer(self.polygon)
        else:
            self.polygon = None

    def _handle_marker_move(self, marker, location, **kwargs):
        old_loc = marker.location
        new_loc = location
        idx = self.domain_coords.index(old_loc)
        self.domain_coords[idx] = new_loc
        self._update_domain_render()

    def _handle_map_click(self, **kwargs):
        if kwargs.get("type") == "click":
            loc = kwargs.get("coordinates")
            marker = Marker(location=loc)
            marker.on_move(partial(self._handle_marker_move, marker=marker))
            self.domain_coords.append(loc)
            self.marker_locs[marker] = loc
            self.m.add_layer(marker)
            self._update_domain_render()

    def _clear_domain(self, *args, **kwargs):
        self.domain_coords = []
        for marker in self.marker_locs.keys():
            self.m.remove_layer(marker)
        self.marker_locs = {}
        self._update_domain_render()

    def _save_domain(self, *args, **kwargs):
        fn = "{}.domain.yaml".format(self.name_textfield.value)
        with open(fn, "w") as fh:
            yaml.dump(self.domain_coords, fh, default_flow_style=False)
        print("Domain points written to `{}`".format(fn))
