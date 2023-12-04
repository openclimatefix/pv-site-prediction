from psp.data_sources.nwp import NwpDataSource


class SatelliteDataSource(NwpDataSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, filter_on_step=False)

    # need to think if we want to tidy this up or have our own class for satellite
