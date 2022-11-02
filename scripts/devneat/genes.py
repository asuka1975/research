from neat.genes import BaseGene
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute


class DevnetNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        FloatAttribute('nx'),
                        FloatAttribute('ny'),
                        StringAttribute('activation', options='sigmoid'),
                        StringAttribute('aggregation', options='sum')]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response) + abs(self.nx - other.nx) + abs(self.ny - other.ny)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


class DevnetConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        FloatAttribute('cx'),
                        FloatAttribute('cy'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight) + abs(self.cx - other.cx) + abs(self.cy - other.cy)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient