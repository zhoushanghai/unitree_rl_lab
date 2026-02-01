// Keyboard control enabled for Sim2Sim testing
#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_Mimic.h"


std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-29dof Controller \n";
    std::cout << " [Keyboard Control Enabled] \n\n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 5; // 29dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "========================================\n";
    std::cout << "        KEYBOARD CONTROL MODE           \n";
    std::cout << "========================================\n";
    std::cout << " State Transitions:\n";
    std::cout << "   [1] -> Enter FixStand (stand up)\n";
    std::cout << "   [2] -> Enter Velocity (RL control)\n";
    std::cout << "   [0] -> Return to Passive (safe)\n";
    std::cout << "----------------------------------------\n";
    std::cout << " Velocity Control (in RL mode):\n";
    std::cout << "   W/S -> Forward/Backward\n";
    std::cout << "   A/D -> Left/Right strafe\n";
    std::cout << "   Q/E -> Rotate left/right\n";
    std::cout << "========================================\n";
    std::cout << " Gamepad also works if connected.\n";
    std::cout << "========================================\n\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

