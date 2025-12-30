// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "AegisAVDataTypes.h"
#include "AegisAVWebSocketClient.generated.h"

class IWebSocket;

/**
 * WebSocket client for connecting to the AegisAV Python backend.
 *
 * Handles:
 * - WebSocket connection lifecycle (connect, disconnect, reconnect)
 * - JSON message parsing and dispatching
 * - Base64 image decoding for camera frames
 * - Event broadcasting via delegates
 *
 * Usage:
 *   UAegisAVWebSocketClient* Client = NewObject<UAegisAVWebSocketClient>();
 *   Client->OnTelemetryReceived.AddDynamic(this, &MyClass::HandleTelemetry);
 *   Client->Connect("ws://localhost:8080/ws/unreal");
 */
UCLASS(BlueprintType)
class AEGISAVOVERLAY_API UAegisAVWebSocketClient : public UObject
{
    GENERATED_BODY()

public:
    UAegisAVWebSocketClient();
    virtual ~UAegisAVWebSocketClient();

    // ========================================================================
    // Connection Management
    // ========================================================================

    /**
     * Connect to the AegisAV WebSocket server
     * @param URL WebSocket URL (default: ws://localhost:8080/ws/unreal)
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|WebSocket")
    void Connect(const FString& URL = TEXT("ws://localhost:8080/ws/unreal"));

    /**
     * Disconnect from the server
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|WebSocket")
    void Disconnect();

    /**
     * Check if currently connected
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|WebSocket")
    bool IsConnected() const;

    /**
     * Get the current server URL
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|WebSocket")
    FString GetServerURL() const { return ServerURL; }

    /**
     * Enable/disable automatic reconnection on disconnect
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|WebSocket")
    void SetAutoReconnect(bool bEnable, float IntervalSeconds = 5.0f);

    // ========================================================================
    // Event Delegates
    // ========================================================================

    /** Fired when telemetry is received */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisTelemetryReceived OnTelemetryReceived;

    /** Fired when agent thinking state is received */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisAgentThoughtReceived OnAgentThoughtReceived;

    /** Fired when critic evaluation is received */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisCriticEvaluationReceived OnCriticEvaluationReceived;

    /** Fired when battery status is received */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisBatteryStatusReceived OnBatteryStatusReceived;

    /** Fired when anomaly is detected */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisAnomalyReceived OnAnomalyReceived;

    /** Fired when camera frame is received (with decoded texture) */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisCameraFrameReceived OnCameraFrameReceived;

    /** Fired when connection state changes */
    UPROPERTY(BlueprintAssignable, Category = "AegisAV|Events")
    FOnAegisConnectionStateChanged OnConnectionStateChanged;

protected:
    // WebSocket instance
    TSharedPtr<IWebSocket> WebSocket;

    // Connection settings
    FString ServerURL;
    bool bAutoReconnect;
    float ReconnectInterval;
    FTimerHandle ReconnectTimerHandle;

    // State
    bool bIsConnected;
    int32 ReconnectAttempts;

    // WebSocket callbacks
    void OnWebSocketConnected();
    void OnWebSocketConnectionError(const FString& Error);
    void OnWebSocketClosed(int32 StatusCode, const FString& Reason, bool bWasClean);
    void OnWebSocketMessage(const FString& Message);
    void OnWebSocketRawMessage(const void* Data, SIZE_T Size, SIZE_T BytesRemaining);

    // Message parsing
    void ParseAndDispatchMessage(const FString& JsonString);
    void ParseTelemetryMessage(const TSharedPtr<class FJsonObject>& JsonObject);
    void ParseAgentThinkingMessage(const TSharedPtr<class FJsonObject>& JsonObject);
    void ParseCriticResultMessage(const TSharedPtr<class FJsonObject>& JsonObject);
    void ParseBatteryUpdateMessage(const TSharedPtr<class FJsonObject>& JsonObject);
    void ParseAnomalyMessage(const TSharedPtr<class FJsonObject>& JsonObject);
    void ParseCameraFrameMessage(const TSharedPtr<class FJsonObject>& JsonObject);

    // Helper functions
    UTexture2D* Base64ToTexture2D(const FString& Base64String, int32 Width, int32 Height);
    ECriticVerdict StringToVerdict(const FString& VerdictString) const;
    ERiskLevel StringToRiskLevel(const FString& RiskString) const;

    // Reconnection
    void AttemptReconnect();
    void StopReconnectTimer();

    // Cached textures for camera frames (to avoid recreating every frame)
    UPROPERTY()
    TMap<FString, UTexture2D*> CameraTextureCache;
};
