import os
import warnings

from neat.config import ConfigParameter, ConfigParser, UnknownConfigItemError, write_pretty_params, Config
from devneat.task import create_task

class CustomConfig(object):
    """A container for user-configurable parameters of NEAT."""

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False),
                ConfigParameter('observe_tick', int),
                ConfigParameter('num_generations', int),
                ConfigParameter('enable_devrule_per_neurocomponents', bool, False),
                ConfigParameter('enable_evaluation_by_complexity', bool, False)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, task_config, filename, config_information=None):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.task_config = task_config
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.config_information = config_information

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            parameters.read_file(f)

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn(f"Using default {p.default!r} for '{p.name!s}'",
                                  DeprecationWarning)
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError(f"Unknown (section 'NEAT') configuration item {unknown_list[0]!s}")

        self.genome_type = genome_type
        # Parse type sections.
        genome_dict = dict(
            CreatorGenome=dict(parameters.items("CreatorGenome")), 
            DeleterGenome=dict(parameters.items("DeleterGenome")),
            SolverGenome=dict(parameters.items("SolverGenome")),
            CustomGenome=dict(parameters.items("CustomGenome"))
        )
        genome_dict["SolverGenome"]["num_inputs"] = max(create_task(self.task_config["tasks"][i]).num_inputs() for i in self.task_config["schedule"])
        genome_dict["SolverGenome"]["num_outputs"] = max(create_task(self.task_config["tasks"][i]).num_outputs() for i in self.task_config["schedule"])
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            f.write('# or the experiment itself.  This is the only required section.\n')
            f.write('[NEAT]\n')
            write_pretty_params(f, self, self.__params)

            f.write(f'\n[{self.genome_type.__name__}]\n')
            self.genome_type.write_config(f, self.genome_config)

            f.write(f'\n[{self.species_set_type.__name__}]\n')
            self.species_set_type.write_config(f, self.species_set_config)

            f.write(f'\n[{self.stagnation_type.__name__}]\n')
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write(f'\n[{self.reproduction_type.__name__}]\n')
            self.reproduction_type.write_config(f, self.reproduction_config)

class CustomStaticConfig(object):
    """A container for user-configurable parameters of NEAT."""

    __params = [ConfigParameter('pop_size', int),
                ConfigParameter('fitness_criterion', str),
                ConfigParameter('fitness_threshold', float),
                ConfigParameter('reset_on_extinction', bool),
                ConfigParameter('no_fitness_termination', bool, False), 
                ConfigParameter('observe_tick', int),
                ConfigParameter('num_generations', int)]

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, task_config, filename, config_information=None):
        # Check that the provided types have the required methods.
        assert hasattr(genome_type, 'parse_config')
        assert hasattr(reproduction_type, 'parse_config')
        assert hasattr(species_set_type, 'parse_config')
        assert hasattr(stagnation_type, 'parse_config')

        self.genome_type = genome_type
        self.reproduction_type = reproduction_type
        self.species_set_type = species_set_type
        self.stagnation_type = stagnation_type
        self.config_information = config_information

        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        parameters = ConfigParser()
        with open(filename) as f:
            parameters.read_file(f)

        # NEAT configuration
        if not parameters.has_section('NEAT'):
            raise RuntimeError("'NEAT' section not found in NEAT configuration file.")

        param_list_names = []
        for p in self.__params:
            if p.default is None:
                setattr(self, p.name, p.parse('NEAT', parameters))
            else:
                try:
                    setattr(self, p.name, p.parse('NEAT', parameters))
                except Exception:
                    setattr(self, p.name, p.default)
                    warnings.warn(f"Using default {p.default!r} for '{p.name!s}'",
                                  DeprecationWarning)
            param_list_names.append(p.name)
        param_dict = dict(parameters.items('NEAT'))
        unknown_list = [x for x in param_dict if x not in param_list_names]
        if unknown_list:
            if len(unknown_list) > 1:
                raise UnknownConfigItemError("Unknown (section 'NEAT') configuration items:\n" + "\n\t".join(unknown_list))
            raise UnknownConfigItemError(f"Unknown (section 'NEAT') configuration item {unknown_list[0]!s}")

        self.task_config = task_config
        # Parse type sections.
        genome_dict = dict(parameters.items(genome_type.__name__))
        genome_dict["num_inputs"] = max(create_task(self.task_config["tasks"][i]).num_inputs() for i in self.task_config["schedule"])
        genome_dict["num_outputs"] = max(create_task(self.task_config["tasks"][i]).num_outputs() for i in self.task_config["schedule"])
        self.genome_config = genome_type.parse_config(genome_dict)

        species_set_dict = dict(parameters.items(species_set_type.__name__))
        self.species_set_config = species_set_type.parse_config(species_set_dict)

        stagnation_dict = dict(parameters.items(stagnation_type.__name__))
        self.stagnation_config = stagnation_type.parse_config(stagnation_dict)

        reproduction_dict = dict(parameters.items(reproduction_type.__name__))
        self.reproduction_config = reproduction_type.parse_config(reproduction_dict)

    def save(self, filename):
        with open(filename, 'w') as f:
            f.write('# The `NEAT` section specifies parameters particular to the NEAT algorithm\n')
            f.write('# or the experiment itself.  This is the only required section.\n')
            f.write('[NEAT]\n')
            write_pretty_params(f, self, self.__params)

            f.write(f'\n[{self.genome_type.__name__}]\n')
            self.genome_type.write_config(f, self.genome_config)

            f.write(f'\n[{self.species_set_type.__name__}]\n')
            self.species_set_type.write_config(f, self.species_set_config)

            f.write(f'\n[{self.stagnation_type.__name__}]\n')
            self.stagnation_type.write_config(f, self.stagnation_config)

            f.write(f'\n[{self.reproduction_type.__name__}]\n')
            self.reproduction_type.write_config(f, self.reproduction_config)