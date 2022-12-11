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
    float flipAmount;
    Quaternion prevRot;

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

        // Get the hinges controlling the arm and the hand
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
        // Start with bottle unreleased and zero bottle flip degrees each episode
        releasedBottle = false;
        flipAmount = 0;

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

        // Initialize the previous rotation of the bottle to the starting rotation of the bottle
        prevRot = bottleRot;

        // Add and get the joint holding the bottle to the hand
        if (bottleJoint == null)
        {
            bottleJoint = bottle.AddComponent<FixedJoint>();
            bottleJoint.connectedBody = handRb;
        }
    }

    // Get observations about the current state (arm hinge vel, hand hinge vel, bottle x pos, bottle y pos)
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(armHinge.angle);      // Current angle of arm relative to joint (degrees)
        sensor.AddObservation(armHinge.velocity);   // Current angular velocity of arm around joint (degrees/second)
        sensor.AddObservation(handHinge.angle);     // Current angle of hand relative to joint (degrees)
        sensor.AddObservation(handHinge.velocity);  // Current angular velocity of hand around joint (degrees/second)
    }

    // Receive actions, utilize them, and apply rewards
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        JointMotor armMotor;
        JointMotor handMotor;

        // If the max number of steps has been reached, end episode and give negative reward
        if (StepCount >= 600)
        {
            SetReward(-10.0f);
            Debug.Log("Reward = -10");
            EndEpisode();
        }

        // Use actions to move the arm/hand if the bottle has not been released yet
        if (!releasedBottle)
        {

            // Change the target velocity of the arm hinge based on the corresponding action
            armMotor = armHinge.motor;
            if (actionBuffers.DiscreteActions[0] == 1)
            {
                armMotor.targetVelocity += 1;
            } else if (actionBuffers.DiscreteActions[0] == 2)
            {
                armMotor.targetVelocity -= 1;
            }
            armHinge.motor = armMotor;

            // Change the target velocity of the hand hinge based on the corresponding action
            handMotor = handHinge.motor;
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

                SetReward((Mathf.Abs(armHinge.velocity) + Mathf.Abs(handHinge.velocity)) / 50 - 1);
                Debug.Log("Reward = " + ((Mathf.Abs(armHinge.velocity) + Mathf.Abs(handHinge.velocity)) / 50 - 1));
            }
        }
        // If the bottle has been released, then just apply rewards for flipping/landing before ending the episode
        else
        {
            // If the bottle is no longer moving after release, end the episode
            if (bottleRb.velocity.magnitude + bottleRb.angularVelocity.magnitude < 0.01)
            {
                float flipReward = Mathf.Abs(flipAmount) / 10;

                // If the bottle lands upright after flipping, give the agent a big reward
                if (Mathf.Abs(flipAmount) > 180 && Vector3.Dot(bottle.transform.up, Vector3.up) > 0.95)
                {
                    SetReward(+100000.0f + flipReward);
                    Debug.Log("Reward = " + (+100000.0f + flipReward));
                }
                // Otherwise, penalize the agent for failing a bottle flip
                else
                {
                    SetReward(-8.0f + flipReward);
                    Debug.Log("Reward = " + (-8.0f + flipReward));
                }
                EndEpisode();
            }
            // Add to the flip amount the difference in degrees from the previous rotation to the current rotation in the XY plane.
            else
            {
                Vector3 curUp = bottle.transform.rotation * Vector3.up;
                Vector3 prevUp = prevRot * Vector3.up;

                float curAng = Mathf.Atan2(curUp.x, curUp.y) * Mathf.Rad2Deg;
                float prevAng = Mathf.Atan2(prevUp.x, prevUp.y) * Mathf.Rad2Deg;

                flipAmount += Mathf.DeltaAngle(curAng, prevAng);
            }
        }

        // Update the previous rotation of the bottle to the current rotation of the bottle
        prevRot = bottle.transform.rotation;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = (int)(Input.GetAxis("Horizontal")) == -1 ? 2 : (int)(Input.GetAxis("Horizontal"));
        discreteActionsOut[1] = (int)(Input.GetAxis("Vertical")) == -1 ? 2 : (int)(Input.GetAxis("Vertical"));
        discreteActionsOut[2] = Input.GetButton("Fire1") ? 1 : 0;
    }
}
