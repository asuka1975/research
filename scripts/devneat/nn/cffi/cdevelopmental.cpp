#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

#include "crecurrent.hpp"

namespace {
    double tanh_activation(double z) {
        z = std::max(-60.0, std::min(60.0, 2.5 * z));
        return std::tanh(z);
    }

    double distance2(double x1, double y1, double x2, double y2) {
        return std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2);
    }

    double energy_curve(double v) {
        return std::log(v) - std::log(1 - v);
    }

    class creator_t {
    public:
        creator_t(RecurrentNetwork& net, int neighbor_num)
            : neighbor_num(neighbor_num), func(net) {
            int num_inputs = 4 * neighbor_num + 3;
            for(int i = 0; i < num_inputs; i++) {
                inputs.append(0);
            }
        }
        std::pair<std::tuple<double, double, double, bool>, std::tuple<double, double, double, bool>> operator()(std::vector<std::ptrdiff_t>::iterator nodes_begin, std::vector<std::ptrdiff_t>::iterator nodes_end, const std::vector<std::tuple<double, double, double, double>>& nodes, std::tuple<double, double, double> conn_avg) {
            int i = 0;
            for(auto iter = nodes_begin; iter != nodes_end; i++, iter++) {
                inputs[4 * i] = std::get<0>(nodes[*iter]);
                inputs[4 * i + 1] = std::get<1>(nodes[*iter]);
                inputs[4 * i + 2] = std::get<2>(nodes[*iter]);
                inputs[4 * i + 3] = std::get<3>(nodes[*iter]);
            }
            inputs[4 * neighbor_num] = std::get<0>(conn_avg);
            inputs[4 * neighbor_num + 1] = std::get<1>(conn_avg);
            inputs[4 * neighbor_num + 2] = std::get<2>(conn_avg);
            auto o = func.get().activate(inputs);
            double nx = boost::python::extract<double>(o[0]);
            double ny = boost::python::extract<double>(o[1]);
            double bias = boost::python::extract<double>(o[2]);
            double nf = boost::python::extract<double>(o[3]);
            double cx = boost::python::extract<double>(o[4]);
            double cy = boost::python::extract<double>(o[5]);
            double weight = boost::python::extract<double>(o[6]);
            double cf = boost::python::extract<double>(o[7]);
            return std::make_pair(std::make_tuple(nx, ny, bias, nf > 0), std::make_tuple(cx, cy, weight, cf > 0));
        }
    private:
        boost::python::list inputs;
        int neighbor_num;
        std::reference_wrapper<RecurrentNetwork> func;
    };

    class deleter_t {
    public:
        deleter_t(RecurrentNetwork& net) 
            : func(net) {
            for(int i = 0; i < 11; i++) {
                inputs.append(0);
            }
        }
        bool operator()(std::tuple<double, double, double, double> in, std::tuple<double, double, double, double> out, std::tuple<double, double, int, int, double> conn) {
            inputs[0] = std::get<0>(in);
            inputs[1] = std::get<1>(in);
            inputs[2] = std::get<2>(in);
            inputs[3] = std::get<3>(in);
            inputs[4] = std::get<0>(out);
            inputs[5] = std::get<1>(out);
            inputs[6] = std::get<2>(out);
            inputs[7] = std::get<3>(out);
            inputs[8] = std::get<0>(conn);
            inputs[9] = std::get<1>(conn);
            inputs[10] = std::get<4>(conn);
            auto o = func.get().activate(inputs);
            return boost::python::extract<double>(o[0]) > 0.0;
        }
    private:
        boost::python::list inputs;
        std::reference_wrapper<RecurrentNetwork> func;
    };
}

class DevelopmentalNetwork {
public:
    DevelopmentalNetwork(int num_inputs, int num_outputs, const boost::python::list& conns, const boost::python::list& nodes, const boost::python::dict& devconfig, boost::python::object& creator, boost::python::object& deleter)
    : num_inputs(num_inputs), num_outputs(num_outputs), active(0), step(1), develop_tick(boost::python::extract<int>(devconfig["num_develop_steps"])), num_neighbors(boost::python::extract<int>(devconfig["num_neighbors"])),
        creator(boost::python::extract<RecurrentNetwork&>(creator), boost::python::extract<int>(devconfig["num_neighbors"])),
        deleter(boost::python::extract<RecurrentNetwork&>(deleter)),
        nodes(boost::python::len(nodes)), conns(boost::python::len(conns)),
        devrule_per_neurocomponents(devconfig.has_key("enable_devrule_per_neurocomponents") ? devconfig["enable_devrule_per_neurocomponents"] : false),
        origin_creator((RecurrentNetwork&)boost::python::extract<RecurrentNetwork&>(creator)), origin_deleter((RecurrentNetwork&)boost::python::extract<RecurrentNetwork&>(deleter)) {
        for(int i = 0; i < boost::python::len(nodes); i++) {
            double x = boost::python::extract<double>(nodes[i][0]);
            double y = boost::python::extract<double>(nodes[i][1]);
            double bias = boost::python::extract<double>(nodes[i][2]);
            this->nodes[i] = std::make_tuple(x, y, bias, 0.5);
        }
        for(int i = 0; i < boost::python::len(conns); i++) {
            double x = boost::python::extract<double>(conns[i][0]);
            double y = boost::python::extract<double>(conns[i][1]);
            int in = boost::python::extract<int>(conns[i][2]);
            int out = boost::python::extract<int>(conns[i][3]);
            double weight = boost::python::extract<double>(conns[i][4]);
            this->conns[i] = std::make_tuple(x, y, in, out, weight);
        }
        for(auto& values_ : values) {
            values_ = std::vector<double>(this->nodes.size());
        }
        for(int i = 0; i < num_outputs; i++) {
            m_outputs.append(0);
        }
        auto hebb_config = devconfig["hebb"];
        std::get<0>(hebb) = boost::python::extract<double>(hebb_config[0]);
        std::get<1>(hebb) = boost::python::extract<double>(hebb_config[1]);
        std::get<2>(hebb) = boost::python::extract<double>(hebb_config[2]);
        std::get<3>(hebb) = boost::python::extract<double>(hebb_config[3]);
        std::get<4>(hebb) = boost::python::extract<double>(hebb_config[4]);
        if(devrule_per_neurocomponents) {
            auto& crnn = (RecurrentNetwork&)boost::python::extract<RecurrentNetwork&>(creator);
            for(int i = 0; i < boost::python::len(nodes); i++) {
                creator_networks.push_back(crnn.clone());
                creators.emplace_back(creator_networks.back(), num_neighbors);
            }
            auto& drnn = (RecurrentNetwork&)boost::python::extract<RecurrentNetwork&>(deleter);
            for(int i = 0; i < boost::python::len(conns); i++) {
                deleter_networks.push_back(drnn.clone());
                deleters.emplace_back(deleter_networks.back());
            }
        }
    }

    boost::python::list activate(const boost::python::list& inputs) {
        int len = boost::python::len(inputs);
        if(num_inputs != len) {
            throw std::runtime_error("invalid inputs");
        }
        auto& ivalues = values[active];
        auto& ovalues = values[1 - active];
        active = 1 - active;

        for(int i = 0; i < len; i++) {
            ivalues[i] = boost::python::extract<double>(inputs[i]);
            ovalues[i] = boost::python::extract<double>(inputs[i]);
        }

        for(auto& c : conns) {
            auto& [x, y, in, out, weight] = c;
            double input = ivalues[in];
            double output = ivalues[in] * weight;
            ovalues[out] += output;
            // update weight
            weight += std::get<0>(hebb) * (std::get<1>(hebb) * input * output + std::get<2>(hebb) * input + std::get<3>(hebb) * output + std::get<4>(hebb));
        }

        for(int i = 0; i < static_cast<int>(nodes.size()); i++) {
            auto& [x, y, bias, energy] = nodes[i];
            // update energy
            double v = energy_curve(energy) + ovalues[i];
            v = std::clamp(v, energy_curve(0.00000001), energy_curve(0.99999999));
            energy = std::clamp(1 / (1 + std::exp(-v)), 0.00000001, 0.99999999);

            ovalues[i] = tanh_activation(ovalues[i] + bias);
        }

        for(int i = num_inputs; i < num_inputs + num_outputs; i++) {
            m_outputs[i - num_inputs] = ovalues[i];
        }

        if((step++) % develop_tick == 0) {
            develop();
        }

        return m_outputs;
    }

    void develop() {
        std::vector<std::ptrdiff_t> removes(conns.size());
        std::iota(removes.begin(), removes.end(), 0);
        std::vector<std::ptrdiff_t>::iterator riter;
        if(devrule_per_neurocomponents) {
            riter = std::remove_if(removes.begin(), removes.end(), [&deleters=this->deleters, &conns=conns, &nodes=nodes](auto i) {
                return !deleters[i](nodes[std::get<2>(conns[i])], nodes[std::get<3>(conns[i])], conns[i]);
            });
        } else {
            riter = std::remove_if(removes.begin(), removes.end(), [&deleter=this->deleter, &conns=conns, &nodes=nodes](auto i) {
                return !deleter(nodes[std::get<2>(conns[i])], nodes[std::get<3>(conns[i])], conns[i]);
            });
        }
        
        removes.erase(riter, removes.end());
        std::sort(removes.begin(), removes.end());

        std::vector<std::ptrdiff_t> indices(nodes.size());
        std::iota(indices.begin(), indices.end(), 0);
        int index = 0;
        for(auto& node : nodes) {
            std::partial_sort(indices.begin(), indices.begin() + num_neighbors, indices.end(), [&nodes=this->nodes, &node] (auto& i, auto& j) {
                return distance2(std::get<0>(node), std::get<1>(node), std::get<0>(nodes[i]), std::get<1>(nodes[i])) > distance2(std::get<0>(node), std::get<1>(node), std::get<0>(nodes[j]), std::get<1>(nodes[j]));
            });
            auto [sum, len] = std::accumulate(conns.begin(), conns.end(), std::make_pair(std::make_tuple(0.0, 0.0, 0.0), 0), [index] (auto& a, auto& b) {
                if(std::get<3>(b) == index) {
                    return std::make_pair(std::make_tuple(std::get<0>(a.first) + std::get<0>(b), std::get<1>(a.first) + std::get<1>(b), std::get<2>(a.first) + std::get<4>(b)), a.second + 1);
                } else {
                    return a;
                }
            });
            std::tuple<double, double, double> conn_avg;
            if(len != 0) {
                std::get<0>(conn_avg) = std::get<0>(sum) / len;
                std::get<1>(conn_avg) = std::get<1>(sum) / len;
                std::get<2>(conn_avg) = std::get<2>(sum) / len;
            }
            std::pair<std::tuple<double, double, double, bool>, std::tuple<double, double, double, bool>> ret;
            if(devrule_per_neurocomponents) {
                ret = creators[index](indices.begin(), indices.begin() + num_neighbors, nodes, conn_avg);
            } else {
                ret = creator(indices.begin(), indices.begin() + num_neighbors, nodes, conn_avg);
            }
            auto [nx, ny, bias, nf] = ret.first;
            auto [cx, cy, weight, cf] = ret.second;
            double ccx = cx, ccy = cy;
            if(nf) {
                nodes.emplace_back(nx, ny, bias, 0.5);
                values[0].push_back(0);
                values[1].push_back(0);
                if(devrule_per_neurocomponents) {
                    creator_networks.push_back(origin_creator.get().clone());
                    creators.emplace_back(creator_networks.back(), num_neighbors);
                }
            }
            if(cf) {
                auto iter = std::min_element(nodes.begin(), nodes.end(), [ccx, ccy](auto& a, auto & b) { 
                    return distance2(ccx, ccy, std::get<0>(a), std::get<1>(a)) < distance2(ccx, ccy, std::get<0>(b), std::get<1>(b));
                });
                conns.emplace_back(cx, cy, index, iter - nodes.begin(), weight);
                if(devrule_per_neurocomponents) {
                    deleter_networks.push_back(origin_deleter.get().clone());
                    deleters.emplace_back(deleter_networks.back());
                }
            }
            index++;
        }
        std::for_each(removes.rbegin(), removes.rend(), [&conns=this->conns, &deleters=this->deleters, &deleter_networks=this->deleter_networks, pf=devrule_per_neurocomponents](auto i) mutable {
            conns.erase(conns.begin() + i);
            if(pf) {
                deleters.erase(deleters.begin() + i);
                deleter_networks.erase(deleter_networks.begin() + i);
            }
        });
    }

    void reset() {
        for(int i = 0; i < 2; i++) {
            for(int j = 0; values[i].size(); j++) {
                values[i][j] = 0;
            }
        }
        active = 0;
    }

    boost::python::list get_conns() {
        boost::python::list cs;
        for(auto [x, y, i, o, w] : conns) {
            cs.append(boost::python::make_tuple(boost::python::make_tuple(i, o), boost::python::make_tuple(x, y), w));
        }
        return cs;
    }

    boost::python::list get_nodes() {
        boost::python::list ns;
        for(auto [x, y, bias, energy] : nodes) {
            ns.append(boost::python::make_tuple(0, 0, bias, 0, boost::python::make_tuple(x, y), energy));
        }
        return ns;
    }
private:
    boost::python::list m_outputs;
    int num_inputs;
    int num_outputs;
    int active;
    int step;
    int develop_tick;
    int num_neighbors;
    std::tuple<double, double, double, double, double> hebb;
    creator_t creator;
    deleter_t deleter;
    std::vector<std::tuple<double, double, double, double>> nodes; // x, y, bias, energy
    std::vector<std::tuple<double, double, int, int, double>> conns; // x, y, in, out, weight
    std::array<std::vector<double>, 2> values;
    bool devrule_per_neurocomponents;
    std::vector<RecurrentNetwork> creator_networks;
    std::vector<RecurrentNetwork> deleter_networks;
    std::vector<creator_t> creators;
    std::vector<deleter_t> deleters;
    std::reference_wrapper<RecurrentNetwork> origin_creator;
    std::reference_wrapper<RecurrentNetwork> origin_deleter;
};

BOOST_PYTHON_MODULE(cdevelopmental)
{
    using namespace boost::python;
    Py_Initialize();

    class_<DevelopmentalNetwork>("DevelopmentalNetwork", init<int, int, boost::python::list, boost::python::list, boost::python::dict, boost::python::object&, boost::python::object&>())
        .def("activate", &DevelopmentalNetwork::activate)
        .def("reset", &DevelopmentalNetwork::reset)
        .add_property("conns", &DevelopmentalNetwork::get_conns)
        .add_property("nodes", &DevelopmentalNetwork::get_nodes);
}