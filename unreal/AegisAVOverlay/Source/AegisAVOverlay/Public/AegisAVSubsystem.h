// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "AegisAVDataTypes.h"
#include "AegisAVSubsystem.generated.h"

class UAegisAVWebSocketClient;
class UAegisAVMasterHUD;

/**
 * AegisAV Game Instance Subsystem
 *
 * Central manager for the AegisAV overlay system. Handles:
 * - WebSocket client lifecycle
 * - HUD widget creation and management
 * - Input bindings for toggling panels
 *
 * This subsystem persists across level loads, maintaining WebSocket connection.
 *
 * Usage (Blueprint):
 *   UAegisAVSubsystem* Subsystem = UGameplayStatics::GetGameInstance(this)->GetSubsystem<UAegisAVSubsystem>();
 *   Subsystem->ConnectToBackend();
 *   Subsystem->ShowOverlay();
 *
 * Usage (C++):
 *   if (UAegisAVSubsystem* Subsystem = GetGameInstance()->GetSubsystem<UAegisAVSubsystem>())
 *   {
 *       Subsystem->ConnectToBackend();
 *   }
 */
UCLASS()
class AEGISAVOVERLAY_API UAegisAVSubsystem : public UGameInstanceSubsystem
{
    GENERATED_BODY()

public:
    // ========================================================================
    // Subsystem Lifecycle
    // ========================================================================

    virtual void Initialize(FSubsystemCollectionBase& Collection) override;
    virtual void Deinitialize() override;

    // ========================================================================
    // Connection Management
    // ========================================================================

    /**
     * Connect to the AegisAV backend server
     * @param URL WebSocket URL (default: ws://localhost:8080/ws/unreal)
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV")
    void ConnectToBackend(const FString& URL = TEXT("ws://localhost:8080/ws/unreal"));

    /**
     * Disconnect from the backend server
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV")
    void DisconnectFromBackend();

    /**
     * Check if connected to backend
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV")
    bool IsConnectedToBackend() const;

    /**
     * Get the WebSocket client
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV")
    UAegisAVWebSocketClient* GetWebSocketClient() const { return WebSocketClient; }

    // ========================================================================
    // Overlay Management
    // ========================================================================

    /**
     * Show the AegisAV overlay HUD
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Overlay")
    void ShowOverlay();

    /**
     * Hide the AegisAV overlay HUD
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Overlay")
    void HideOverlay();

    /**
     * Toggle overlay visibility
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Overlay")
    void ToggleOverlay();

    /**
     * Check if overlay is currently visible
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Overlay")
    bool IsOverlayVisible() const;

    /**
     * Get the Master HUD widget
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Overlay")
    UAegisAVMasterHUD* GetMasterHUD() const { return MasterHUD; }

    // ========================================================================
    // Panel Management
    // ========================================================================

    /**
     * Show a specific panel by name
     * Valid names: ThoughtBubble, Camera, Critics, Telemetry, Battery
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Panels")
    void ShowPanel(FName PanelName);

    /**
     * Hide a specific panel by name
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Panels")
    void HidePanel(FName PanelName);

    /**
     * Toggle a specific panel by name
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Panels")
    void TogglePanel(FName PanelName);

    // ========================================================================
    // Configuration
    // ========================================================================

    /** The widget class to use for the Master HUD */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    TSubclassOf<UAegisAVMasterHUD> MasterHUDClass;

    /** Auto-connect to backend on subsystem initialization */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    bool bAutoConnectOnInit = false;

    /** Auto-show overlay when first player controller is created */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    bool bAutoShowOverlay = true;

    /** Default server URL */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Config")
    FString DefaultServerURL = TEXT("ws://localhost:8080/ws/unreal");

protected:
    // ========================================================================
    // Internal State
    // ========================================================================

    UPROPERTY()
    UAegisAVWebSocketClient* WebSocketClient;

    UPROPERTY()
    UAegisAVMasterHUD* MasterHUD;

    bool bOverlayVisible;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    void CreateMasterHUD();
    void DestroyMasterHUD();
    void BindInputActions();

    UFUNCTION()
    void OnConnectionStateChanged(bool bIsConnected);
};
