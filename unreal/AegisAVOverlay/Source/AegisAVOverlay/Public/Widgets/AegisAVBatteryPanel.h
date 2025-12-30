// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVDataTypes.h"
#include "AegisAVBatteryPanel.generated.h"

class UTextBlock;
class UProgressBar;
class UImage;

/**
 * Battery Panel - displays drone battery status
 *
 * Shows:
 * - Battery percentage with visual bar
 * - Voltage, current, temperature
 * - Estimated time and distance remaining
 * - Warning indicators for low/critical battery
 * - Charging status
 */
UCLASS(Blueprintable)
class AEGISAVOVERLAY_API UAegisAVBatteryPanel : public UAegisAVBasePanel
{
    GENERATED_BODY()

public:
    UAegisAVBatteryPanel(const FObjectInitializer& ObjectInitializer);

    /**
     * Update the panel with new battery status
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Battery")
    void UpdateBatteryStatus(const FAegisBatteryStatus& BatteryStatus);

    /**
     * Get the current battery status
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Battery")
    FAegisBatteryStatus GetCurrentBatteryStatus() const { return CurrentBatteryStatus; }

protected:
    // Main battery display
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UProgressBar* BatteryBar;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* BatteryPercentText;

    // Detail values
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* VoltageText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* CurrentText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* TemperatureText;

    // Estimates
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* TimeRemainingText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* DistanceRemainingText;

    // Status indicators
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* WarningIcon;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* ChargingIcon;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* StatusText;

    // Configuration
    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Battery")
    float LowBatteryThreshold = 30.0f;

    UPROPERTY(EditDefaultsOnly, Category = "AegisAV|Battery")
    float CriticalBatteryThreshold = 15.0f;

    // Blueprint events
    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|Battery")
    void OnBatteryStatusUpdated(const FAegisBatteryStatus& BatteryStatus);

    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|Battery")
    void OnBatteryLow();

    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|Battery")
    void OnBatteryCritical();

    virtual void NativeConstruct() override;

private:
    FAegisBatteryStatus CurrentBatteryStatus;
    bool bWasLow = false;
    bool bWasCritical = false;

    FLinearColor GetBatteryColor(float Percent, bool bIsCharging) const;
    FString FormatTimeRemaining(float Seconds) const;
};
