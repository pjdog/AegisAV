// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVCriticPanel.h"
#include "Components/Image.h"
#include "Components/TextBlock.h"
#include "Components/ProgressBar.h"

UAegisAVCriticPanel::UAegisAVCriticPanel(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    PanelTitle = FText::FromString(TEXT("Critic Evaluation"));
    ExpandedSize = FVector2D(300.0f, 200.0f);
}

void UAegisAVCriticPanel::NativeConstruct()
{
    Super::NativeConstruct();

    // Initialize with empty state
    ClearEvaluation();
}

void UAegisAVCriticPanel::UpdateEvaluation(const FAegisCriticEvaluation& Evaluation)
{
    CurrentEvaluation = Evaluation;

    // Update safety critic
    UpdateCriticDisplay(SafetyIndicator, SafetyVerdictText, SafetyConfidenceBar, Evaluation.SafetyCritic);

    // Update efficiency critic
    UpdateCriticDisplay(EfficiencyIndicator, EfficiencyVerdictText, EfficiencyConfidenceBar, Evaluation.EfficiencyCritic);

    // Update goal alignment critic
    UpdateCriticDisplay(GoalIndicator, GoalVerdictText, GoalConfidenceBar, Evaluation.GoalAlignmentCritic);

    // Notify Blueprint
    OnEvaluationUpdated(Evaluation);
}

void UAegisAVCriticPanel::ClearEvaluation()
{
    CurrentEvaluation = FAegisCriticEvaluation();

    // Reset all displays to neutral state
    FLinearColor NeutralColor = FLinearColor::Gray;

    if (SafetyIndicator) SafetyIndicator->SetColorAndOpacity(NeutralColor);
    if (SafetyVerdictText) SafetyVerdictText->SetText(FText::FromString(TEXT("-")));
    if (SafetyConfidenceBar) SafetyConfidenceBar->SetPercent(0.0f);

    if (EfficiencyIndicator) EfficiencyIndicator->SetColorAndOpacity(NeutralColor);
    if (EfficiencyVerdictText) EfficiencyVerdictText->SetText(FText::FromString(TEXT("-")));
    if (EfficiencyConfidenceBar) EfficiencyConfidenceBar->SetPercent(0.0f);

    if (GoalIndicator) GoalIndicator->SetColorAndOpacity(NeutralColor);
    if (GoalVerdictText) GoalVerdictText->SetText(FText::FromString(TEXT("-")));
    if (GoalConfidenceBar) GoalConfidenceBar->SetPercent(0.0f);
}

void UAegisAVCriticPanel::UpdateCriticDisplay(UImage* Indicator, UTextBlock* VerdictText, UProgressBar* ConfidenceBar,
                                              const FAegisCriticResult& Result)
{
    FLinearColor VerdictColor = GetVerdictColor(Result.Verdict);
    FString VerdictStr = GetVerdictString(Result.Verdict);

    if (Indicator)
    {
        Indicator->SetColorAndOpacity(VerdictColor);
    }

    if (VerdictText)
    {
        VerdictText->SetText(FText::FromString(VerdictStr));
        VerdictText->SetColorAndOpacity(FSlateColor(VerdictColor));
    }

    if (ConfidenceBar)
    {
        ConfidenceBar->SetPercent(Result.Confidence);
        ConfidenceBar->SetFillColorAndOpacity(VerdictColor);
    }
}

FLinearColor UAegisAVCriticPanel::GetVerdictColor(ECriticVerdict Verdict) const
{
    switch (Verdict)
    {
    case ECriticVerdict::Approve:
        return FLinearColor::Green;

    case ECriticVerdict::ApproveWithConcerns:
        return FLinearColor::Yellow;

    case ECriticVerdict::Escalate:
        return FLinearColor(1.0f, 0.5f, 0.0f); // Orange

    case ECriticVerdict::Reject:
        return FLinearColor::Red;

    default:
        return FLinearColor::Gray;
    }
}

FString UAegisAVCriticPanel::GetVerdictString(ECriticVerdict Verdict) const
{
    switch (Verdict)
    {
    case ECriticVerdict::Approve:
        return TEXT("APPROVE");

    case ECriticVerdict::ApproveWithConcerns:
        return TEXT("CONCERNS");

    case ECriticVerdict::Escalate:
        return TEXT("ESCALATE");

    case ECriticVerdict::Reject:
        return TEXT("REJECT");

    default:
        return TEXT("-");
    }
}
