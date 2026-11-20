package liteoff;

import org.car2x.veins.base.connectionManager.ConnectionManager;
import org.car2x.veins.base.modules.BaseWorldUtility;
import org.car2x.veins.modules.mobility.traci.TraCIScenarioManagerForker;
import org.car2x.veins.modules.obstacle.ObstacleControl;
import org.car2x.veins.modules.world.annotations.AnnotationManager;
import org.car2x.veins.nodes.Car;
import org.car2x.veins.nodes.RSU;

// SUMO is forked by the manager on a free port, so concurrent runs on one node never collide.
network LiteOffScenario
{
    parameters:
        double playgroundSizeX @unit(m);
        double playgroundSizeY @unit(m);
        double playgroundSizeZ @unit(m) = default(50m);
        int numRsu = default(0);
        @display("bgb=dropUnit($playgroundSizeX),dropUnit($playgroundSizeY)");
    submodules:
        world: LiteOffWorld;
        obstacles: ObstacleControl;
        annotations: AnnotationManager;
        connectionManager: ConnectionManager;
        worldUtility: BaseWorldUtility {
            playgroundSizeX = parent.playgroundSizeX;
            playgroundSizeY = parent.playgroundSizeY;
            playgroundSizeZ = parent.playgroundSizeZ;
        }
        manager: TraCIScenarioManagerForker;
        rsu[numRsu]: RSU;
        node[0]: Car;
    connections allowunconnected:
}
