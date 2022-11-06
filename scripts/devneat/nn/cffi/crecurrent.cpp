#include "crecurrent.hpp"
#include <algorithm>
#include <map>
#include <limits>
#include <stdexcept>

namespace activations {
    double sigmoid_activation(double z) {
        z = std::max(-60.0, std::min(60.0, 5.0 * z));
        return 1.0 / (1.0 + std::exp(-z));
    }

    double tanh_activation(double z) {
        z = std::max(-60.0, std::min(60.0, 2.5 * z));
        return std::tanh(z);
    }

    double sin_activation(double z) {
        z = std::max(-60.0, std::min(60.0, 5.0 * z));
        return std::sin(z);
    }

    double gauss_activation(double z) {
        z = std::max(-3.4, std::min(3.4, z));
        return std::exp(-5.0 * z * z);
    }

    double relu_activation(double z) {
        return z > 0.0 ? z : 0.0;
    }

    double elu_activation(double z) {
        return z > 0.0 ? z : std::exp(z) - 1;
    }

    double lelu_activation(double z) {
        double leaky = 0.005;
        return z > 0.0 ? z :leaky * z;
    }

    double selu_activation(double z) {
        double lam = 1.0507009873554804934193349852946;
        double alpha = 1.6732632423543772848170429916717;
        return  z > 0.0 ? lam * z : lam * alpha * (std::exp(z) - 1);
    }

    double softplus_activation(double z) {
        z = std::max(-60.0, std::min(60.0, 5.0 * z));
        return 0.2 * std::log(1 + std::exp(z));
    }

    double identity_activation(double z) {
        return z;
    }

    double clamped_activation(double z) {
        return std::max(-1.0, std::min(1.0, z));
    }

    double inv_activation(double z) {
        if(std::abs(z) < std::numeric_limits<double>::epsilon()) {
            return 0.0;
        } else {
            return 1.0 / z;
        }
    }

    double log_activation(double z) {
        z = std::max(1e-7, z);
        return std::log(z);
    }

    double exp_activation(double z) {
        z = std::max(-60.0, std::min(60.0, z));
        return std::exp(z);
    }

    double abs_activation(double z) {
        return std::abs(z);
    }

    double hat_activation(double z) {
        return std::max(0.0, 1 - std::abs(z));
    }

    double square_activation(double z) {
        return z * z;
    }

    double cube_activation(double z) {
        return z * z * z;
    }
}

std::map<std::string, std::function<double(double)>> activation_functions = {
    { "sigmoid", activations::sigmoid_activation },
    { "tanh", activations::tanh_activation },
    { "sin", activations::sin_activation },
    { "gauss", activations::gauss_activation },
    { "relu", activations::relu_activation },
    { "elu", activations::elu_activation },
    { "lelu", activations::lelu_activation },
    { "selu", activations::selu_activation },
    { "softplus", activations::softplus_activation },
    { "identity", activations::identity_activation },
    { "clamped", activations::clamped_activation },
    { "inv", activations::inv_activation },
    { "log", activations::log_activation },
    { "exp", activations::exp_activation },
    { "abs", activations::abs_activation },
    { "hat", activations::hat_activation },
    { "square", activations::square_activation },
    { "cube", activations::cube_activation }
};

RecurrentNetwork::RecurrentNetwork(int num_inputs, int num_outputs, const boost::python::list& nodes, const boost::python::list& conns, const boost::python::list& activation_defs)
    : num_inputs(num_inputs), num_outputs(num_outputs), 
    nodes(boost::python::len(nodes)), conns(boost::python::len(conns)), activations(boost::python::len(activation_defs))
        {
    for(int i = 0; i < boost::python::len(activation_defs); i++) {
        const char* func_name = boost::python::extract<const char*>(activation_defs[i]);
        activations[i] = activation_functions[func_name];
    }
    for(int i = 0; i < boost::python::len(nodes); i++) {
        double bias = boost::python::extract<double>(nodes[i][0]);
        int activation_id = boost::python::extract<int>(nodes[i][1]);
        this->nodes[i] = std::make_tuple(bias, activation_id);
    }
    for(int i = 0; i < boost::python::len(conns); i++) {
        int in = boost::python::extract<int>(conns[i][0]);
        int out = boost::python::extract<int>(conns[i][1]);
        double weight = boost::python::extract<double>(conns[i][2]);
        this->conns[i] = std::make_tuple(in, out, weight);
    }
    active = 0;
    for(auto& values_ : values) {
        values_ = std::vector<double>(this->nodes.size());
    }
    for(int i = 0; i < num_outputs; i++) {
        m_outputs.append(0);
    }
}

boost::python::list RecurrentNetwork::activate(const boost::python::list& inputs) {
    int len = boost::python::len(inputs);
    if(num_inputs != len) {
        throw std::runtime_error("invalid inputs");
    }
    auto& ivalues = values[active];
    auto& ovalues = values[1 - active];
    active = 1 - active;

    for(int i = 0; i < len; i++) {
        ivalues[i] = boost::python::extract<double>(inputs[i]);
        ovalues[i] = 0;
    }

    for(auto& c : conns) {
        auto& [in, out, weight] = c;
        ovalues[out] += ivalues[in] * weight;
    }

    for(int i = 0; i < static_cast<int>(nodes.size()); i++) {
        auto& [bias, activation_id] = nodes[i];
        ovalues[i] = activations[activation_id](ovalues[i] + bias);
    }

    for(int i = num_inputs; i < num_inputs + num_outputs; i++) {
        m_outputs[i - num_inputs] = ovalues[i];
    }

    return m_outputs;
}

void RecurrentNetwork::reset() {
    for(int i = 0; i < 2; i++) {
        for(int j = 0; values[i].size(); j++) {
            values[i][j] = 0;
        }
    }
    active = 0;
}

RecurrentNetwork RecurrentNetwork::clone() const noexcept {
    RecurrentNetwork rnn;
    rnn.num_inputs = num_inputs;
    rnn.num_outputs = num_outputs;
    rnn.active = 0;
    rnn.nodes = nodes;
    rnn.conns = conns;
    rnn.activations = activations;
    for(auto& values_ : rnn.values) {
        values_ = std::vector<double>(nodes.size());
    }
    for(int i = 0; i < num_outputs; i++) {
        rnn.m_outputs.append(0);
    }
    return rnn;
}

boost::python::list RecurrentNetwork::get_conns() {
    boost::python::list cs;
    for(auto [i, o, w] : conns) {
        cs.append(boost::python::make_tuple(boost::python::make_tuple(i, o), w));
    }
    return cs;
}

boost::python::list RecurrentNetwork::get_nodes() {
    boost::python::list ns;
    for(auto [bias, fid] : nodes) {
        ns.append(boost::python::make_tuple(bias, fid));
    }
    return ns;
}

BOOST_PYTHON_MODULE(crecurrent)
{
    using namespace boost::python;
    Py_Initialize();
    class_<RecurrentNetwork>("RecurrentNetwork", init<int, int, boost::python::list, boost::python::list, boost::python::list>())
        .def("activate", &RecurrentNetwork::activate)
        .def("reset", &RecurrentNetwork::reset)
        .add_property("conns", &RecurrentNetwork::get_conns)
        .add_property("nodes", &RecurrentNetwork::get_nodes);
}