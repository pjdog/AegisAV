// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "AegisAVDataTypes.generated.h"

/**
 * Message types received from the AegisAV WebSocket server
 */
UENUM(BlueprintType)
enum class EAegisMessageType : uint8
{
    Unknown         UMETA(DisplayName = "Unknown"),
    Telemetry       UMETA(DisplayName = "Telemetry"),
    AgentThinking   UMETA(DisplayName = "Agent Thinking"),
    CriticResult    UMETA(DisplayName = "Critic Result"),
    CameraFrame     UMETA(DisplayName = "Camera Frame"),
    AnomalyDetected UMETA(DisplayName = "Anomaly Detected"),
    BatteryUpdate   UMETA(DisplayName = "Battery Update"),
    Decision        UMETA(DisplayName = "Decision"),
    RiskUpdate      UMETA(DisplayName = "Risk Update")
};

/**
 * Verdict from critic evaluation
 */
UENUM(BlueprintType)
enum class ECriticVerdict : uint8
{
    Approve             UMETA(DisplayName = "Approve"),
    ApproveWithConcerns UMETA(DisplayName = "Approve with Concerns"),
    Escalate            UMETA(DisplayName = "Escalate"),
    Reject              UMETA(DisplayName = "Reject")
};

/**
 * Risk level classification
 */
UENUM(BlueprintType)
enum class ERiskLevel : uint8
{
    Low      UMETA(DisplayName = "Low"),
    Medium   UMETA(DisplayName = "Medium"),
    High     UMETA(DisplayName = "High"),
    Critical UMETA(DisplayName = "Critical")
};

/**
 * Telemetry data from drone
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisTelemetry
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    FVector Position = FVector::ZeroVector;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    FVector Velocity = FVector::ZeroVector;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    FRotator Attitude = FRotator::ZeroRotator;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    float BatteryPercent = 100.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    float AltitudeM = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    float SpeedMs = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Telemetry")
    float Timestamp = 0.0f;
};

/**
 * Agent thinking state - shows what the AI is currently considering
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisAgentThought
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString CognitiveLevel;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString Urgency;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString CurrentGoal;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString TargetAsset;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString Situation;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    TArray<FString> Considerations;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    TArray<FString> Options;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    FString SelectedDecision;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    float Confidence = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    float RiskScore = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    ERiskLevel RiskLevel = ERiskLevel::Low;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Thinking")
    float Timestamp = 0.0f;
};

/**
 * Single critic evaluation result
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisCriticResult
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    FString CriticName;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    ECriticVerdict Verdict = ECriticVerdict::Approve;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    float Confidence = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    TArray<FString> Concerns;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    float ProcessingTimeMs = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    bool bUsedLLM = false;
};

/**
 * Full critic evaluation with all three critics
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisCriticEvaluation
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    FAegisCriticResult SafetyCritic;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    FAegisCriticResult EfficiencyCritic;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    FAegisCriticResult GoalAlignmentCritic;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Critic")
    float Timestamp = 0.0f;
};

/**
 * Battery status update
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisBatteryStatus
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float Percent = 100.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float Voltage = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float Current = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float TemperatureC = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float TimeRemainingS = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float DistanceRemainingM = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    bool bIsCharging = false;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    bool bIsCritical = false;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    bool bIsLow = false;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Battery")
    float Timestamp = 0.0f;
};

/**
 * Anomaly detection event
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisAnomaly
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    FString AnomalyId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    FString AnomalyType;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    FString Description;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    float Severity = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    FVector Location = FVector::ZeroVector;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Anomaly")
    float Timestamp = 0.0f;
};

/**
 * Asset spawn data for world placement
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisAssetSpawn
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString AssetId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString AssetType;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString Name;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    double Latitude = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    double Longitude = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    double AltitudeM = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    int32 Priority = 1;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    bool bHasAnomaly = false;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    float AnomalySeverity = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    float Scale = 1.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    float RotationDeg = 0.0f;
};

/**
 * Anomaly marker spawn data for world placement
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisAnomalyMarker
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString AnomalyId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString AssetId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    float Severity = 0.0f;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    double Latitude = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    double Longitude = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    double AltitudeM = 0.0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString MarkerType;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FLinearColor Color = FLinearColor::Red;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    bool bPulse = false;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Assets")
    FString Label;
};

/**
 * Camera frame data (base64 decoded separately)
 */
USTRUCT(BlueprintType)
struct AEGISAVOVERLAY_API FAegisCameraFrame
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Camera")
    FString DroneId;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Camera")
    int32 Sequence = 0;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Camera")
    int32 Width = 1280;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Camera")
    int32 Height = 720;

    UPROPERTY(BlueprintReadOnly, Category = "AegisAV|Camera")
    float TimestampMs = 0.0f;

    // Note: Image data is handled separately via texture
};

// ============================================================================
// Delegates for event broadcasting
// ============================================================================

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisTelemetryReceived, const FAegisTelemetry&, Telemetry);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisAgentThoughtReceived, const FAegisAgentThought&, Thought);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisCriticEvaluationReceived, const FAegisCriticEvaluation&, Evaluation);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisBatteryStatusReceived, const FAegisBatteryStatus&, BatteryStatus);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisAnomalyReceived, const FAegisAnomaly&, Anomaly);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisAssetSpawnReceived, const FAegisAssetSpawn&, Asset);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnAegisAssetsCleared);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisAnomalyMarkerReceived, const FAegisAnomalyMarker&, Marker);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(FOnAegisAnomalyMarkersCleared);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnAegisCameraFrameReceived, const FString&, DroneId, UTexture2D*, Frame);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAegisConnectionStateChanged, bool, bIsConnected);
