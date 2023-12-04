from psp.data_sources.nwp import NwpDataSource


class SatelliteDataSource(NwpDataSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

