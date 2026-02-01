#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

/**
 * @brief FSMState with keyboard support for state transitions
 * 
 * Keyboard Mapping (for Sim2Sim without gamepad):
 * ------------------------------------------------
 * Key '1' -> Enter FixStand mode (replaces L2 + Up)
 * Key '2' -> Enter Velocity/RL mode (replaces R1 + X)
 * Key '0' -> Return to Passive mode (replaces L2 + B)
 */
class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // ============ KEYBOARD-BASED TRANSITIONS ============
        // Key '1' -> FixStand
        if(FSMStringMap.right.count("FixStand")) {
            int fixstand_id = FSMStringMap.right.at("FixStand");
            registered_checks.emplace_back(
                std::make_pair(
                    []()->bool{ 
                        return keyboard && keyboard->key() == "1" && keyboard->on_pressed; 
                    },
                    fixstand_id
                )
            );
        }

        // Key '2' -> Velocity (RL mode)
        if(FSMStringMap.right.count("Velocity")) {
            int velocity_id = FSMStringMap.right.at("Velocity");
            registered_checks.emplace_back(
                std::make_pair(
                    []()->bool{ 
                        return keyboard && keyboard->key() == "2" && keyboard->on_pressed; 
                    },
                    velocity_id
                )
            );
        }

        // Key '0' -> Passive (safety return)
        if(FSMStringMap.right.count("Passive")) {
            int passive_id = FSMStringMap.right.at("Passive");
            registered_checks.emplace_back(
                std::make_pair(
                    []()->bool{ 
                        return keyboard && keyboard->key() == "0" && keyboard->on_pressed;
                    },
                    passive_id
                )
            );
        }

        // Timeout -> Passive (safety fallback)
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};