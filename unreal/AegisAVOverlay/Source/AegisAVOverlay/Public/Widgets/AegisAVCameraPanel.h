// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVCameraPanel.generated.h"

class UImage;
class UTextBlock;
class UMaterialInstanceDynamic;

/**
 * Camera Panel - displays drone camera feed
 *
 * Receives camera frames as textures from the WebSocket client
 * and displays them in a UImage widget with dynamic material.
 */
UCLASS(Blueprintable)
class AEGISAVOVERLAY_API UAegisAVCameraPanel : public UAegisAVBasePanel
{
    GENERATED_BODY()

public:
    UAegisAVCameraPanel(const FObjectInitializer& ObjectInitializer);

    /**
     * Update the camera feed with a new frame
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Camera")
    void UpdateCameraFrame(UTexture2D* NewFrame);

    /**
     * Set the camera label (e.g., drone ID)
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Camera")
    void SetCameraLabel(const FString& Label);

    /**
     * Get the current frame texture
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Camera")
    UTexture2D* GetCurrentFrame() const { return CurrentFrame; }

protected:
    // Widget bindings
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* CameraImage;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* CameraLabel;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* FrameInfoText;

    // Material for dynamic texture rendering
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Camera")
    UMaterialInterface* CameraMaterial;

    // Blueprint events
    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|Camera")
    void OnCameraFrameUpdated(UTexture2D* Frame);

    virtual void NativeConstruct() override;

private:
    UPROPERTY()
    UTexture2D* CurrentFrame;

    UPROPERTY()
    UMaterialInstanceDynamic* DynamicMaterial;

    int32 FrameCount;
    float LastFrameTime;

    void CreateDynamicMaterial();
};
