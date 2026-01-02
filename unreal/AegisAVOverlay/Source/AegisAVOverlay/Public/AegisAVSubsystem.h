// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/GameInstanceSubsystem.h"
#include "AegisAVDataTypes.h"
#include "AegisAVSubsystem.generated.h"

class UAegisAVWebSocketClient;
class UAegisAVMasterHUD;
class UStaticMesh;
class AActor;

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
     * @param URL WebSocket URL (default: ws://localhost:8090/ws/unreal)
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV")
    void ConnectToBackend(const FString& URL = TEXT("ws://localhost:8090/ws/unreal"));

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
    FString DefaultServerURL = TEXT("ws://localhost:8090/ws/unreal");

    // ========================================================================
    // Asset Spawning Configuration
    // ========================================================================

    /** Enable spawning assets in the Unreal scene */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    bool bEnableAssetSpawning = true;

    /** Use the first received asset as the geo origin */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    bool bUseFirstAssetAsOrigin = true;

    /** Geo origin (degrees/meters). If zero and bUseFirstAssetAsOrigin is true, origin is auto-set. */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    double OriginLatitude = 0.0;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    double OriginLongitude = 0.0;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    double OriginAltitude = 0.0;

    /** Unreal units per meter (default: 100cm per meter) */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    float UnitsPerMeter = 100.0f;

    /** Mesh paths for asset types (SoftObjectPath strings) */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FString DefaultAssetMeshPath = TEXT("/Engine/BasicShapes/Cube.Cube");

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FString SolarPanelMeshPath = TEXT("/Engine/BasicShapes/Plane.Plane");

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FString WindTurbineMeshPath = TEXT("/Engine/BasicShapes/Cone.Cone");

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FString SubstationMeshPath = TEXT("/Engine/BasicShapes/Cube.Cube");

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FString PowerLineMeshPath = TEXT("/Engine/BasicShapes/Cylinder.Cylinder");

    /** Default per-type scale */
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FVector DefaultAssetScale = FVector(1.0f, 1.0f, 1.0f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FVector SolarPanelScale = FVector(6.0f, 3.0f, 0.2f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FVector WindTurbineScale = FVector(1.0f, 1.0f, 6.0f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FVector SubstationScale = FVector(2.5f, 2.5f, 1.5f);

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Assets")
    FVector PowerLineScale = FVector(0.2f, 0.2f, 5.0f);

protected:
    // ========================================================================
    // Internal State
    // ========================================================================

    UPROPERTY()
    UAegisAVWebSocketClient* WebSocketClient;

    UPROPERTY()
    UAegisAVMasterHUD* MasterHUD;

    UPROPERTY()
    TMap<FString, TWeakObjectPtr<AActor>> SpawnedAssets;

    UPROPERTY()
    TMap<FString, TWeakObjectPtr<AActor>> SpawnedAnomalyMarkers;

    bool bOverlayVisible;
    bool bOriginInitialized = false;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    void CreateMasterHUD();
    void DestroyMasterHUD();
    void BindInputActions();
    void LoadConfigFromFile();
    void BindWebSocketEvents();

    UFUNCTION()
    void OnAssetSpawnReceived(const FAegisAssetSpawn& Asset);

    UFUNCTION()
    void OnAssetsCleared();

    UFUNCTION()
    void OnAnomalyMarkerReceived(const FAegisAnomalyMarker& Marker);

    UFUNCTION()
    void OnAnomalyMarkersCleared();

    FVector GeoToWorld(double Latitude, double Longitude, double AltitudeM) const;
    UStaticMesh* LoadMeshForAssetType(const FString& AssetType) const;
    FVector GetScaleForAssetType(const FString& AssetType) const;
    void InitializeOriginIfNeeded(const FAegisAssetSpawn& Asset);
    AActor* SpawnMeshActor(const FString& AssetId, const FString& AssetType, const FString& Name);
    void ClearSpawnedActors(TMap<FString, TWeakObjectPtr<AActor>>& ActorMap);

    UFUNCTION()
    void OnConnectionStateChanged(bool bIsConnected);
};
