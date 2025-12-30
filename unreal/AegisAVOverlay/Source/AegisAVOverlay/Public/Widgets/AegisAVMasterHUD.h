// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "AegisAVDataTypes.h"
#include "AegisAVMasterHUD.generated.h"

class UAegisAVWebSocketClient;
class UAegisAVBasePanel;
class UCanvasPanel;
class UTextBlock;
class UImage;

/**
 * Master HUD container for all AegisAV overlay panels.
 *
 * Manages:
 * - Panel visibility and positioning
 * - WebSocket event routing to panels
 * - Connection status indicator
 * - Layout save/load
 *
 * The MasterHUD is created by UAegisAVSubsystem and added to the viewport.
 * It contains a Canvas Panel that holds all individual draggable panels.
 */
UCLASS(Blueprintable)
class AEGISAVOVERLAY_API UAegisAVMasterHUD : public UUserWidget
{
    GENERATED_BODY()

public:
    // ========================================================================
    // Initialization
    // ========================================================================

    /**
     * Initialize the HUD with a WebSocket client for data
     * @param InWebSocketClient The client to receive data from
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void Initialize(UAegisAVWebSocketClient* InWebSocketClient);

    // ========================================================================
    // Panel Management
    // ========================================================================

    /**
     * Show a specific panel
     * @param PanelName Name of the panel (ThoughtBubble, Camera, Critics, Telemetry, Battery)
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void ShowPanel(FName PanelName);

    /**
     * Hide a specific panel
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void HidePanel(FName PanelName);

    /**
     * Toggle panel visibility
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void TogglePanel(FName PanelName);

    /**
     * Check if a panel is visible
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|HUD")
    bool IsPanelVisible(FName PanelName) const;

    /**
     * Show all panels
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void ShowAllPanels();

    /**
     * Hide all panels
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void HideAllPanels();

    // ========================================================================
    // Connection Status
    // ========================================================================

    /**
     * Update the connection status indicator
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void SetConnectionState(bool bIsConnected);

    // ========================================================================
    // Layout Management
    // ========================================================================

    /**
     * Save current panel positions to config
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void SaveLayout();

    /**
     * Load panel positions from config
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void LoadLayout();

    /**
     * Reset panels to default positions
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|HUD")
    void ResetLayout();

    // ========================================================================
    // Widget Bindings
    // ========================================================================

protected:
    /** Main canvas that contains all panels */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidget))
    UCanvasPanel* MainCanvas;

    /** Connection status indicator */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* ConnectionIndicator;

    /** Connection status text */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* ConnectionText;

    // ========================================================================
    // Panel Widgets (optional - can be created dynamically)
    // ========================================================================

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UAegisAVBasePanel* ThoughtBubblePanel;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UAegisAVBasePanel* CameraPanel;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UAegisAVBasePanel* CriticsPanel;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UAegisAVBasePanel* TelemetryPanel;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UAegisAVBasePanel* BatteryPanel;

    // ========================================================================
    // Configuration
    // ========================================================================

    /** Classes for dynamically created panels */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    TSubclassOf<UAegisAVBasePanel> ThoughtBubblePanelClass;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    TSubclassOf<UAegisAVBasePanel> CameraPanelClass;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    TSubclassOf<UAegisAVBasePanel> CriticsPanelClass;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    TSubclassOf<UAegisAVBasePanel> TelemetryPanelClass;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    TSubclassOf<UAegisAVBasePanel> BatteryPanelClass;

    /** Default positions for panels */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    FVector2D ThoughtBubbleDefaultPos = FVector2D(20.0f, 20.0f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    FVector2D CameraDefaultPos = FVector2D(20.0f, 300.0f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    FVector2D CriticsDefaultPos = FVector2D(400.0f, 20.0f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    FVector2D TelemetryDefaultPos = FVector2D(400.0f, 300.0f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    FVector2D BatteryDefaultPos = FVector2D(780.0f, 20.0f);

    // ========================================================================
    // Overrides
    // ========================================================================

    virtual void NativeConstruct() override;
    virtual void NativeDestruct() override;

private:
    // ========================================================================
    // Internal State
    // ========================================================================

    UPROPERTY()
    UAegisAVWebSocketClient* WebSocketClient;

    bool bIsConnected;

    // Map panel names to widgets
    TMap<FName, UAegisAVBasePanel*> PanelMap;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    void SetupPanelMap();
    void CreateDynamicPanels();
    void BindWebSocketEvents();
    void UnbindWebSocketEvents();
    UAegisAVBasePanel* GetPanelByName(FName PanelName) const;

    // ========================================================================
    // WebSocket Event Handlers
    // ========================================================================

    UFUNCTION()
    void OnTelemetryReceived(const FAegisTelemetry& Telemetry);

    UFUNCTION()
    void OnAgentThoughtReceived(const FAegisAgentThought& Thought);

    UFUNCTION()
    void OnCriticEvaluationReceived(const FAegisCriticEvaluation& Evaluation);

    UFUNCTION()
    void OnBatteryStatusReceived(const FAegisBatteryStatus& BatteryStatus);

    UFUNCTION()
    void OnAnomalyReceived(const FAegisAnomaly& Anomaly);

    UFUNCTION()
    void OnCameraFrameReceived(const FString& DroneId, UTexture2D* Frame);

    UFUNCTION()
    void OnConnectionStateChanged(bool bConnected);
};
