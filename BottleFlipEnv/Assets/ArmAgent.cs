using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class ArmAgent : Agent
{
    GameObject arm;
    GameObject hand;
    GameObject bottle;

    Rigidbody armRb;
    Rigidbody handRb;
    Rigidbody bottleRb;

    HingeJoint armHinge;
    HingeJoint handHinge;
    FixedJoint bottleJoint;

    Vector3 armPos;
    Vector3 handPos;
    Vector3 bottlePos;

    Quaternion armRot;
    Quaternion handRot;
    Quaternion bottleRot;

    bool releasedBottle;
    int flipCount;
    bool canCountFlips;

    void Start()
    {
        // Get the GameObject for the arm, hand, and bottle
        arm = GameObject.Find("Arm");
        hand = GameObject.Find("Hand");
        bottle = GameObject.Find("Bottle");

        // Get the Rigidbody for the arm, hand, and bottle
        armRb = arm.GetComponent<Rigidbody>();
        handRb = hand.GetComponent<Rigidbody>();
        bottleRb = bottle.GetComponent<Rigidbody>();

        // Get the hinges controlling the arm and hand
        armHinge = arm.GetComponent<HingeJoint>();
        handHinge = hand.GetComponent<HingeJoint>();

        // Get the starting position and rotation for the arm, hand, and bottle
        armPos = arm.transform.position;
        armRot = arm.transform.rotation;
        handPos = hand.transform.position;
        handRot = hand.transform.rotation;
        bottlePos = bottle.transform.position;
        bottleRot = bottle.transform.rotation;
    }

    // Reset scene environment setup for episode start
    public override void OnEpisodeBegin()
    {
        // Start with bottle unreleased and zero bottle flips each episode
        releasedBottle = false;
        flipCount = 0;
        canCountFlips = true;

        // Initialize/reset the arm and hand joints to have target velocity of zero
        JointMotor armMotor = armHinge.motor;
        armMotor.force = 1000000;
        armMotor.targetVelocity = 0;
        armMotor.freeSpin = false;
        armHinge.motor = armMotor;

        JointMotor handMotor = handHinge.motor;
        handMotor.force = 1000000;
        handMotor.targetVelocity = 0;
        handMotor.freeSpin = false;
        handHinge.motor = handMotor;

        // Initialize/reset the velocities of the arm, hand, and bottle
        armRb.velocity = Vector3.zero;
        armRb.angularVelocity = Vector3.zero;
        handRb.velocity = Vector3.zero;
        handRb.angularVelocity = Vector3.zero;
        bottleRb.velocity = Vector3.zero;
        bottleRb.angularVelocity = Vector3.zero;

        // Initialize/reset the position and rotation of the arm, hand, and bottle
        arm.transform.position = armPos;
        arm.transform.rotation = armRot;
        hand.transform.position = handPos;
        hand.transform.rotation = handRot;
        bottle.transform.position = bottlePos;
        bottle.transform.rotation = bottleRot;

        // Add and get the joint holding the bottle to the hand
        bottleJoint = bottle.AddComponent<FixedJoint>();
        bottleJoint.connectedBody = handRb;
    }

    // Get observations about the current state (arm hinge vel, hand hinge vel, bottle x pos, bottle y pos)
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(armHinge.motor.targetVelocity);       // Current target velocity of arm
        sensor.AddObservation(handHinge.motor.targetVelocity);      // Current target velocity of hand
        sensor.AddObservation(bottle.transform.localPosition.x);    // Current x location of bottle
        sensor.AddObservation(bottle.transform.localPosition.y);    // Current y location of bottle
    }

    // Receive actions, utilize them, and apply rewards
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        // If the max number of steps has been reached, end episode and give negative reward
        if (StepCount >= 500)
        {
            SetReward(-20.0f);
            Debug.Log("Reward = -20");
            EndEpisode();
        }

        // Use actions to move the arm/hand if the bottle has not been released yet
        if (!releasedBottle)
        {

            // Change the target velocity of the arm hinge based on the corresponding action
            JointMotor armMotor = armHinge.motor;
            if (actionBuffers.DiscreteActions[0] == 1)
            {
                armMotor.targetVelocity += 1;
            } else if (actionBuffers.DiscreteActions[0] == 2)
            {
                armMotor.targetVelocity -= 1;
            }
            armHinge.motor = armMotor;

            // Change the target velocity of the hand hinge based on the corresponding action
            JointMotor handMotor = handHinge.motor;
            if (actionBuffers.DiscreteActions[1] == 1)
            {
                handMotor.targetVelocity += 1;
            }
            else if (actionBuffers.DiscreteActions[1] == 2)
            {
                handMotor.targetVelocity -= 1;
            }
            handHinge.motor = handMotor;

            // Release the bottle if the agent decides to
            if (actionBuffers.DiscreteActions[2] == 1)
            {
                Destroy(bottleJoint);
                releasedBottle = true;
            }
        }
        // If the bottle has been released, then just apply rewards for flipping/landing before ending the episode
        else
        {
            // If the bottle is no longer moving after release, end the episode
            if (bottleRb.velocity.magnitude + bottleRb.angularVelocity.magnitude < 0.01)
            {
                // If the bottle lands upright after flipping, give the agent a big reward
                if (flipCount > 0 && Vector3.Dot(bottle.transform.up, Vector3.up) > 0.95)
                {
                    SetReward(+100.0f);
                    Debug.Log("Reward = +100");
                }
                // Otherwise, penalize the agent for failing a bottle flip
                else
                {
                    SetReward(-10.0f);
                    Debug.Log("Reward = -10");
                }
                EndEpisode();
            }
            // If the bottle is upside down and the flip counter is enabled, add to the flip count and disable the flip counter to avoid double counting
            else if (canCountFlips && Vector3.Dot(bottle.transform.up, Vector3.down) > 0.95)
            {
                flipCount++;
                Debug.Log("Reward = +5");
                SetReward(+5.0f);
                canCountFlips = false;
            }
            // If the bottle is right side up, re-enable the flip counter
            else if (!canCountFlips && Vector3.Dot(bottle.transform.up, Vector3.up) > 0.95)
            {
                canCountFlips = true;
            }
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = (int)(Input.GetAxis("Horizontal")) == -1 ? 2 : (int)(Input.GetAxis("Horizontal"));
        discreteActionsOut[1] = (int)(Input.GetAxis("Vertical")) == -1 ? 2 : (int)(Input.GetAxis("Vertical"));
        discreteActionsOut[2] = Input.GetButton("Fire1") ? 1 : 0;
    }
}
