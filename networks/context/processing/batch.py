from collections import OrderedDict

from networks.context.debug import DebugKeys

from sample import Sample


class MiniBatch(object):
    """
    Is a neural network batch that is consist of bags.
    """

    def __init__(self, bags, batch_id=None):
        assert(isinstance(batch_id, int) or batch_id is None)
        assert(isinstance(bags, list))
        self._batch_id = batch_id
        self.bags = bags

    def to_network_input(self):
        result = OrderedDict()

        for sample in self.iter_by_samples():

            assert(isinstance(sample, Sample))

            for arg, value in sample:
                if arg not in result:
                    result[arg] = []
                result[arg].append(value)

        if DebugKeys.MiniBatchShow:
            MiniBatch.debug_output(result)

        return result

    @staticmethod
    def debug_output(result):
        print "-------------------"
        for key, value in result.iteritems():
            print "{}: {}".format(key, value)
        print "-------------------"

    def iter_by_samples(self):
        for bag in self.bags:
            for sample in bag:
                yield sample

    def __iter__(self):
        for bag in self.bags:
            yield bag
