// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVDataTypes.h"
#include "AegisAVThoughtBubblePanel.generated.h"

class UTextBlock;
class UVerticalBox;
class UProgressBar;
class URichTextBlock;

/**
 * Thought Bubble Panel - displays agent thinking state
 *
 * Shows:
 * - Current cognitive level and urgency
 * - Active goal and target
 * - Situation assessment
 * - List of considerations
 * - Available options
 * - Selected decision with confidence
 * - Risk score indicator
 */
UCLASS(Blueprintable)
class AEGISAVOVERLAY_API UAegisAVThoughtBubblePanel : public UAegisAVBasePanel
{
    GENERATED_BODY()

public:
    UAegisAVThoughtBubblePanel(const FObjectInitializer& ObjectInitializer);

    /**
     * Update the panel with new thinking data
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|ThoughtBubble")
    void UpdateThought(const FAegisAgentThought& Thought);

    /**
     * Get the current thought state
     */
    UFUNCTION(BlueprintPure, Category = "AegisAV|ThoughtBubble")
    FAegisAgentThought GetCurrentThought() const { return CurrentThought; }

    /**
     * Clear the thought display
     */
    UFUNCTION(BlueprintCallable, Category = "AegisAV|ThoughtBubble")
    void ClearThought();

protected:
    // Widget bindings
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* CognitiveLevelText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* UrgencyText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* GoalText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* TargetText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* SituationText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UVerticalBox* ConsiderationsBox;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UVerticalBox* OptionsBox;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* DecisionText;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UProgressBar* ConfidenceBar;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UProgressBar* RiskBar;

    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* RiskLevelText;

    // Blueprint events
    UFUNCTION(BlueprintImplementableEvent, Category = "AegisAV|ThoughtBubble")
    void OnThoughtUpdated(const FAegisAgentThought& Thought);

    virtual void NativeConstruct() override;

private:
    FAegisAgentThought CurrentThought;

    void UpdateConsiderationsDisplay();
    void UpdateOptionsDisplay();
    FLinearColor GetRiskLevelColor(ERiskLevel Level) const;
    FLinearColor GetUrgencyColor(const FString& Urgency) const;
};
