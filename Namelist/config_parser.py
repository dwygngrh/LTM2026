import configparser
from datetime import datetime

class LTMConfig:
    def __init__(self, file_path="namelist.txt"):
        self.config = configparser.ConfigParser()
        self.config.read(file_path)

    def get_simulation_times(self):
        start = datetime.strptime(self.config.get('Simulation', 'start_simulation'), '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(self.config.get('Simulation', 'end_simulation'), '%Y-%m-%d %H:%M:%S')
        return start, end

    def get_sources(self):
        sources = []
        for section in self.config.sections():
            if section.startswith('Source'):
                src = {
                    "name": self.config.get(section, 'name'),
                    "start_rel": datetime.strptime(self.config.get(section, 'start_date_release'), '%Y-%m-%d %H:%M:%S'),
                    "end_rel": datetime.strptime(self.config.get(section, 'end_date_release'), '%Y-%m-%d %H:%M:%S'),
                    "lon": self.config.getfloat(section, 'lon'),
                    "lat": self.config.getfloat(section, 'lat'),
                    "depth": self.config.getfloat(section, 'depth'),
                    "diameter": self.config.getfloat(section, 'diameter'),
                    "type": self.config.get(section, 'type'),
                    "rate": self.config.getint(section, 'release_rate')
                }
                sources.append(src)
        return sources
