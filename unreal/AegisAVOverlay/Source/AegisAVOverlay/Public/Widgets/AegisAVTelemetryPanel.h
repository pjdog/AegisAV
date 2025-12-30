// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVDataTypes.h"
#include "AegisAVTelemetryPanel.generated.h"

class UTextBlock;

/**
 * Telemetry Panel - displays drone position, velocity, and attitude
 *
 * Shows:
 * - Position (X, Y, Z)
 * - Velocity (X, Y, Z)
 * - Attitude (Pitch, Roll, Yaw)
 * - Altitude
 * - Speed
 */
UCLASS(Blueprintable)
class AEGISAVOVERLAY_API UAegisAVTelemetryPanel : public UAegisAVBasePanel
{
    GENERATED_BODY()

public:
    UAegisAVTelemetryPanel(const FObjectInitializer& ObjectInitializer);

    /**
     * Update the panel with new telemetry data
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Telemetry")
    void UpdateTelemetry(const FAegisTelemetry& Telemetry);

    /**
     * Get the current telemetry
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Telemetry")
    FAegisTelemetry GetCurrentTelemetry() const { return CurrentTelemetry; }

protected:
    // Position widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* PositionXText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* PositionYText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* PositionZText;

    // Velocity widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* VelocityXText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* VelocityYText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* VelocityZText;

    // Attitude widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* PitchText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* RollText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* YawText;

    // Summary widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* AltitudeText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* SpeedText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* DroneIdText;

    // Blueprint events
    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|Telemetry")
    void OnTelemetryUpdated(const FAegisTelemetry& Telemetry);

    virtual void NativeConstruct() override;

private:
    FAegisTelemetry CurrentTelemetry;
};
