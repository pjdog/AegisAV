// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVDataTypes.h"
#include "AegisAVCriticPanel.generated.h"

class UImage;
class UTextBlock;
class UProgressBar;

/**
 * Critic Panel - displays the three critic evaluations
 *
 * Shows:
 * - Safety critic verdict and confidence
 * - Efficiency critic verdict and confidence
 * - Goal alignment critic verdict and confidence
 *
 * Each critic has a color-coded indicator:
 * - Green: Approve
 * - Yellow: Approve with concerns
 * - Orange: Escalate
 * - Red: Reject
 */
UCLASS(Blueprintable)
class AEGISAVOVERLAY_API UAegisAVCriticPanel : public UAegisAVBasePanel
{
    GENERATED_BODY()

public:
    UAegisAVCriticPanel(const FObjectInitializer& ObjectInitializer);

    /**
     * Update the panel with new critic evaluation
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Critics")
    void UpdateEvaluation(const FAegisCriticEvaluation& Evaluation);

    /**
     * Get the current evaluation
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Critics")
    FAegisCriticEvaluation GetCurrentEvaluation() const { return CurrentEvaluation; }

    /**
     * Clear the evaluation display
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|Critics")
    void ClearEvaluation();

protected:
    // Safety critic widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* SafetyIndicator;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* SafetyVerdictText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UProgressBar* SafetyConfidenceBar;

    // Efficiency critic widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* EfficiencyIndicator;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* EfficiencyVerdictText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UProgressBar* EfficiencyConfidenceBar;

    // Goal alignment critic widgets
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UImage* GoalIndicator;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* GoalVerdictText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UProgressBar* GoalConfidenceBar;

    // Blueprint events
    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|Critics")
    void OnEvaluationUpdated(const FAegisCriticEvaluation& Evaluation);

    virtual void NativeConstruct() override;

    /**
     * Get color for a verdict
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Critics")
    FLinearColor GetVerdictColor(ECriticVerdict Verdict) const;

    /**
     * Get display string for a verdict
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|Critics")
    FString GetVerdictString(ECriticVerdict Verdict) const;

private:
    FAegisCriticEvaluation CurrentEvaluation;

    void UpdateCriticDisplay(UImage* Indicator, UTextBlock* VerdictText, UProgressBar* ConfidenceBar,
                             const FAegisCriticResult& Result);
};
