/**
 *   @file  gtrackUnitStart.c
 *
 *   @brief
 *      Unit level start function for the GTRACK Algorithm
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2017 Texas Instruments, Inc.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>

#include <ti/alg/gtrack/gtrack.h>
#include <ti/alg/gtrack/include/gtrack_int.h>


extern const float eye4[16];
extern const float eye6[36];

/**
*  @b Description
*  @n
*		GTRACK Module calls this function to start target tracking. This function is called during modules' allocation step, 
*		once new set of points passes allocation thresholds 
*
*  @param[in]  handle
*		This is handle to GTRACK unit
*  @param[in]  timeStamp
*		This is an allocation time stamp
*  @param[in]  um
*		This is a pointer to the parental centroid
*  @param[in]  vid
*		Target (vehicle) identifier. Used for debugging purposes only
*
*  \ingroup GTRACK_ALG_UNIT_FUNCTION
*
*  @retval
*      None
*/

void gtrack_unitStart(void *handle, uint64_t timeStamp, float *um)
{
    GtrackUnitInstance *inst;
	float m[3];
	MATRIX6x6 P = {0};

    inst = (GtrackUnitInstance *)handle;
	
	inst->heartBeatCount = timeStamp;
	inst->allocationTime = timeStamp;
	inst->allocationRange = um[0];
	inst->allocationVelocity = um[2];
	inst->vid = 0;

	inst->state = TRACK_STATE_DETECTION;
	inst->currentStateVectorType = GTRACK_STATE_VECTORS_2DA;
	inst->stateVectorLength = 6;

	// initialize process variance to 1/2 of maximum target acceleration
	inst->processVariance = (0.5f*inst->maxAcceleration)*(0.5f*inst->maxAcceleration);

	inst->F = inst->F6;
	inst->Q = inst->Q6;

	// Radial Velocity initialization
	// Radial Velocity handling is set to start with range rate filtering
	inst->velocityHandling = VELOCITY_INIT;

	m[2] = gtrack_unrollRadialVelocity(inst->maxURadialVelocity, inst->initialRadialVelocity, um[2]);

	inst->rangeRate = m[2];

	m[0] = um[0]; // Range
	m[1] = um[1]; // Azimuth

	// Initialize a-priori State information
	gtrack_spherical2cartesian(inst->currentStateVectorType, m, inst->S_apriori_hat);
	memcpy(inst->H_s, m, sizeof(m)); // Initialize Hs to measurment vector
	// Initialize a-priori Process covariance to an identity
	// memcpy(inst->P_apriori_hat, eye6, sizeof(eye6));

	// P_apriori_hat = diag([0,0,0,5,0,5,1,1]);
	P.e33 = P.e44 = 0.5;	
	P.e55 = P.e66 = 1;
	memcpy(inst->P_apriori_hat, &P, sizeof(P));

#ifdef DebugP_LOG_ENABLED
	if(inst->verbose & VERBOSE_WARNING_INFO)
		gtrack_log(GTRACK_VERBOSE_DEBUG, "%llu: tid[%d] ALLOCATED, Range %5.2f, Angle %2.4f, Doppler %5.1f=>%5.1f\n", inst->heartBeatCount, inst->tid, m[0], m[1], um[2], m[2]);
#endif
}
