package liteoff;

import org.car2x.veins.modules.application.ieee80211p.DemoBaseApplLayer;

// Common parameters of every LiteOff node (vehicle or RSU).
simple LiteOffNodeApp extends DemoBaseApplLayer
{
    parameters:
        @class(liteoff::LiteOffNodeApp);
        sendBeacons = false;          // LiteOff sends its own LoBeacon
        dataOnSch = false;
        dataUserPriority = default(4);
        double cpuHz = default(2e9);
        double beaconBytes = default(200);
        double headerBytes = default(40);
        double chunkBytes = default(2000);      // app-level fragment size (<= 2304 B 802.11 MSDU)
        double maxBeaconAge @unit(s) = default(2.5s);
        bool adaptiveMcs = default(true);       // distance-based MCS (6..18 Mbps)
        bool dropExpired = default(true);       // servers skip jobs past their deadline
        double radioPowerW = default(1.0);      // radio electronics power while transmitting
        double kappa = default(1e-28);          // effective switched capacitance (E = k c f^2)
        double commRange @unit(m) = default(300m);
        double macEfficiency = default(0.6);
        double linkMargin = default(1.2);
        double backhaulHopDelay @unit(s) = default(0.005s);
}

simple VehicleApp extends LiteOffNodeApp
{
    parameters:
        @class(liteoff::VehicleApp);
        cpuHz = default(-1);                 // -1: draw uniform(cpuHzMin, cpuHzMax) from the workload RNG
        double cpuHzMin = default(1e9);
        double cpuHzMax = default(3e9);
        double taskRate @unit(Hz) = default(1Hz);
        double taskStopTime @unit(s) = default(1e9s);
        double energyBudgetJ = default(5000);
        bool routeInBeacons = default(true);
        int uplinkBurst = default(2);          // chunks handed to the MAC per pacing slot
        double uplinkPacing = default(1.0);    // pacing slot = burst * airtime * uplinkPacing
        double deadlineScale = default(1.0);
        double selfishProb = default(0.0);     // E9: probability that this vehicle silently drops tasks it should serve
        // task classes (see docs/DEVIATIONS.md for the scaling w.r.t. the plan)
        double pSafety = default(0.5);
        double pPerception = default(0.35);
        double safetyBytesMin = default(20e3);
        double safetyBytesMax = default(100e3);
        double safetyCyclesMin = default(0.05e9);
        double safetyCyclesMax = default(0.2e9);
        double safetyDeadlineMin @unit(s) = default(0.05s);
        double safetyDeadlineMax @unit(s) = default(0.15s);
        double safetyResultBytes = default(200);
        double percBytesMin = default(100e3);
        double percBytesMax = default(500e3);
        double percCyclesMin = default(0.5e9);
        double percCyclesMax = default(2e9);
        double percDeadlineMin @unit(s) = default(0.3s);
        double percDeadlineMax @unit(s) = default(0.5s);
        double percResultBytes = default(2000);
        double infoBytesMin = default(200e3);
        double infoBytesMax = default(1e6);
        double infoCyclesMin = default(0.2e9);
        double infoCyclesMax = default(1e9);
        double infoDeadlineMin @unit(s) = default(1s);
        double infoDeadlineMax @unit(s) = default(2s);
        double infoResultBytes = default(10000);
}

simple RsuApp extends LiteOffNodeApp
{
    parameters:
        @class(liteoff::RsuApp);
        cpuHz = default(20e9);
        double backhaulRate @unit(bps) = default(1Gbps);
}

// Global singleton: scheme configuration, metrics, side-car connection, logs.
simple LiteOffWorld
{
    parameters:
        @class(liteoff::LiteOffWorld);
        @display("i=misc/globe");
        // ---- scheme selection (one decision interface)
        string scheme = default("rule");   // local|rsu|random|rule|basic-rule|oracle|explore|liteoff|b1|b2|b3|b4|b5
        string deploymentMode = default("hybrid");  // vehicle|rsu|hybrid  (liteoff only: modes A/B/C)
        int K = default(5);
        bool gateEnabled = default(true);
        bool cacheEnabled = default(true);
        double cacheTtl @unit(s) = default(2s);
        bool validatorEnabled = default(true);
        double confThreshold = default(0.5);
        double tinyCycles = default(5e7);
        double localMargin = default(0.7);
        string contextFamilies = default("mobility,link,resources,task,trust,road");
        string llmModel = default("liteoff-student-q4");
        string llmLatencyKey = default("");     // latency model row ("" = llmModel)
        string rsuLlmModel = default("qwen2.5:3b");
        string rsuLlmLatencyKey = default("");
        bool ignoreLlmLatency = default(false); // "large model on node (idealised)" ablation
        bool llmOnCpu = default(false);         // false: on-board accelerator queue
        double vehicleLlmSpeedup = default(1.0);
        double rsuLlmSpeedup = default(4.0);
        double llmMaxBudgetFraction = default(0.5);
        bool rsuEscalateCritical = default(false);
        bool reoffloadEnabled = default(true);
        double watchdogFactor = default(2.0);
        double minWatchdog @unit(s) = default(0.05s);
        double lateGrace @unit(s) = default(1s);
        bool cloudEnabled = default(true);
        double cloudHz = default(100e9);
        double cloudDelay @unit(s) = default(0.05s);   // one-way backhaul RSU <-> cloud
        double queryBytesFactor = default(0.5);        // bytes of an RSU LLM query per byte of JSON context
        // ---- exploration (dataset generation)
        double exploreOracleProb = default(0.5);
        double exploreRandomProb = default(0.25);
        // ---- RL baselines
        string rlMode = default("eval");   // train|eval
        string rlWeights = default("");
        double fedPeriod @unit(s) = default(10s);
        double fedModelBytes = default(50000);
        // ---- model lifecycle
        bool modelManagement = default(true);
        double initialModelProb = default(1.0);
        double adapterBytes = default(4e6);
        double modelUpdateInterval @unit(s) = default(0s);   // 0: no new versions during the run
        double modelPushPeriod @unit(s) = default(0.05s);
        double pushCbrThreshold = default(0.4);
        int chunksPerPush = default(2);
        // ---- infrastructure
        string roadType = default("urban");
        string latencyFile = default("");
        string sidecarHost = default("127.0.0.1");
        int sidecarPort = default(0);
        double sidecarConnectTimeout = default(60);
        double warmup @unit(s) = default(0s);
        string scenarioTag = default("");
        string traceFile = default("");
        string datasetFile = default("");
}
