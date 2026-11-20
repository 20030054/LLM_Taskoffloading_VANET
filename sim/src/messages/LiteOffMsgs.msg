//
// LiteOff-LLM application messages (all carried as IEEE 1609.4 WSMs).
//
import veins.modules.messages.BaseFrame1609_4;

namespace liteoff;

// Periodic beacon: mobility, resources, model version and model-download progress.
packet LoBeacon extends veins::BaseFrame1609_4 {
    int src;
    bool isRsu;
    int seq;
    double x;
    double y;
    double vx;
    double vy;
    double cpuHz;
    double backlogSec;
    int queueLen;
    double load;
    double energy;
    double cbr;
    int modelVersion;
    int modelRecvVersion;
    int modelRecvChunks;
    string nextEdges;
}

// One chunk of an offloaded task part.
packet LoTaskChunk extends veins::BaseFrame1609_4 {
    int src;
    long taskId;
    int attempt;
    int origin;
    int seq;
    int nChunks;
    double partBytes;
    double cycles;
    double resultBytes;
    simtime_t created;
    simtime_t deadline;
    bool toCloud;
}

// Result of an executed task part, returned to the origin.
packet LoResult extends veins::BaseFrame1609_4 {
    int src;
    long taskId;
    int attempt;
    int origin;
    int executor;
    int executorKind;
    simtime_t rxDone;     // server: last chunk received
    simtime_t execStart;  // server: execution started
    simtime_t execEnd;    // server: execution finished
}

// Model-distribution chunk (adapter update) or federated-learning model exchange.
packet LoModelChunk extends veins::BaseFrame1609_4 {
    int src;
    int version;
    int seq;
    int nChunks;
    bool federated;
}

// Vehicle -> RSU request for an RSU-hosted LLM decision (deployment modes B/C, baselines B2/B3).
// The structured context itself is kept in the LiteOffWorld side store (simulation shortcut);
// the frame carries its true byte size so channel cost is modelled.
packet LoLlmQuery extends veins::BaseFrame1609_4 {
    int src;
    long queryId;
}

packet LoLlmReply extends veins::BaseFrame1609_4 {
    int src;
    long queryId;
    int slot;
    int split;
    double conf;
    bool ok;
    int nIn;
    int nOut;
}
