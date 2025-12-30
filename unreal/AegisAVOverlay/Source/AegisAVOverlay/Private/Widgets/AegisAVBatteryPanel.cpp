// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVBatteryPanel.h"
#include "Components/TextBlock.h"
#include "Components/ProgressBar.h"
#include "Components/Image.h"

UAegisAVBatteryPanel::UAegisAVBatteryPanel(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    PanelTitle = FText::FromString(TEXT("Battery"));
    ExpandedSize = FVector2D(250.0f, 180.0f);
    MinimizedSize = FVector2D(150.0f, 40.0f);
}

void UAegisAVBatteryPanel::NativeConstruct()
{
    Super::NativeConstruct();

    // Initialize with default values
    CurrentBatteryStatus = FAegisBatteryStatus();
    CurrentBatteryStatus.Percent = 100.0f;
    UpdateBatteryStatus(CurrentBatteryStatus);
}

void UAegisAVBatteryPanel::UpdateBatteryStatus(const FAegisBatteryStatus& BatteryStatus)
{
    CurrentBatteryStatus = BatteryStatus;

    // Update battery bar
    if (BatteryBar)
    {
        BatteryBar->SetPercent(BatteryStatus.Percent / 100.0f);
        BatteryBar->SetFillColorAndOpacity(GetBatteryColor(BatteryStatus.Percent, BatteryStatus.bIsCharging));
    }

    // Update percentage text
    if (BatteryPercentText)
    {
        BatteryPercentText->SetText(FText::FromString(FString::Printf(TEXT("%.0f%%"), BatteryStatus.Percent)));

        // Color code the text
        FLinearColor TextColor = GetBatteryColor(BatteryStatus.Percent, BatteryStatus.bIsCharging);
        BatteryPercentText->SetColorAndOpacity(FSlateColor(TextColor));
    }

    // Update voltage
    if (VoltageText)
    {
        VoltageText->SetText(FText::FromString(FString::Printf(TEXT("%.2f V"), BatteryStatus.Voltage)));
    }

    // Update current
    if (CurrentText)
    {
        FString CurrentStr = FString::Printf(TEXT("%.2f A"), FMath::Abs(BatteryStatus.Current));
        if (BatteryStatus.Current < 0)
        {
            CurrentStr = TEXT("-") + CurrentStr; // Discharging
        }
        CurrentText->SetText(FText::FromString(CurrentStr));
    }

    // Update temperature
    if (TemperatureText)
    {
        TemperatureText->SetText(FText::FromString(FString::Printf(TEXT("%.1f C"), BatteryStatus.TemperatureC)));

        // Warn if temperature is high
        FLinearColor TempColor = FLinearColor::White;
        if (BatteryStatus.TemperatureC > 45.0f)
        {
            TempColor = FLinearColor::Red;
        }
        else if (BatteryStatus.TemperatureC > 40.0f)
        {
            TempColor = FLinearColor::Yellow;
        }
        TemperatureText->SetColorAndOpacity(FSlateColor(TempColor));
    }

    // Update time remaining
    if (TimeRemainingText)
    {
        TimeRemainingText->SetText(FText::FromString(FormatTimeRemaining(BatteryStatus.TimeRemainingS)));
    }

    // Update distance remaining
    if (DistanceRemainingText)
    {
        FString DistStr;
        if (BatteryStatus.DistanceRemainingM >= 1000.0f)
        {
            DistStr = FString::Printf(TEXT("%.1f km"), BatteryStatus.DistanceRemainingM / 1000.0f);
        }
        else
        {
            DistStr = FString::Printf(TEXT("%.0f m"), BatteryStatus.DistanceRemainingM);
        }
        DistanceRemainingText->SetText(FText::FromString(DistStr));
    }

    // Update warning icon
    if (WarningIcon)
    {
        bool bShowWarning = BatteryStatus.bIsLow || BatteryStatus.bIsCritical;
        WarningIcon->SetVisibility(bShowWarning ? ESlateVisibility::Visible : ESlateVisibility::Collapsed);

        if (bShowWarning)
        {
            FLinearColor WarningColor = BatteryStatus.bIsCritical ? FLinearColor::Red : FLinearColor::Yellow;
            WarningIcon->SetColorAndOpacity(WarningColor);
        }
    }

    // Update charging icon
    if (ChargingIcon)
    {
        ChargingIcon->SetVisibility(BatteryStatus.bIsCharging ? ESlateVisibility::Visible : ESlateVisibility::Collapsed);
    }

    // Update status text
    if (StatusText)
    {
        FString Status;
        if (BatteryStatus.bIsCharging)
        {
            Status = TEXT("CHARGING");
        }
        else if (BatteryStatus.bIsCritical)
        {
            Status = TEXT("CRITICAL");
        }
        else if (BatteryStatus.bIsLow)
        {
            Status = TEXT("LOW");
        }
        else
        {
            Status = TEXT("OK");
        }
        StatusText->SetText(FText::FromString(Status));
        StatusText->SetColorAndOpacity(FSlateColor(GetBatteryColor(BatteryStatus.Percent, BatteryStatus.bIsCharging)));
    }

    // Fire events for state transitions
    if (BatteryStatus.bIsCritical && !bWasCritical)
    {
        OnBatteryCritical();
    }
    else if (BatteryStatus.bIsLow && !bWasLow)
    {
        OnBatteryLow();
    }

    bWasLow = BatteryStatus.bIsLow;
    bWasCritical = BatteryStatus.bIsCritical;

    // Notify Blueprint
    OnBatteryStatusUpdated(BatteryStatus);
}

FLinearColor UAegisAVBatteryPanel::GetBatteryColor(float Percent, bool bIsCharging) const
{
    if (bIsCharging)
    {
        return FLinearColor(0.0f, 0.8f, 1.0f); // Cyan for charging
    }

    if (Percent <= CriticalBatteryThreshold)
    {
        return FLinearColor::Red;
    }
    else if (Percent <= LowBatteryThreshold)
    {
        return FLinearColor(1.0f, 0.5f, 0.0f); // Orange
    }
    else if (Percent <= 50.0f)
    {
        return FLinearColor::Yellow;
    }
    else
    {
        return FLinearColor::Green;
    }
}

FString UAegisAVBatteryPanel::FormatTimeRemaining(float Seconds) const
{
    if (Seconds <= 0.0f)
    {
        return TEXT("--:--");
    }

    int32 TotalSeconds = FMath::FloorToInt(Seconds);
    int32 Hours = TotalSeconds / 3600;
    int32 Minutes = (TotalSeconds % 3600) / 60;
    int32 Secs = TotalSeconds % 60;

    if (Hours > 0)
    {
        return FString::Printf(TEXT("%d:%02d:%02d"), Hours, Minutes, Secs);
    }
    else
    {
        return FString::Printf(TEXT("%d:%02d"), Minutes, Secs);
    }
}
