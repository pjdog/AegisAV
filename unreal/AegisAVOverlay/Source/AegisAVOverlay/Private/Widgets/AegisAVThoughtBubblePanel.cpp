// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVThoughtBubblePanel.h"
#include "Components/TextBlock.h"
#include "Components/VerticalBox.h"
#include "Components/ProgressBar.h"

UAegisAVThoughtBubblePanel::UAegisAVThoughtBubblePanel(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    PanelTitle = FText::FromString(TEXT("Agent Thinking"));
    ExpandedSize = FVector2D(400.0f, 350.0f);
}

void UAegisAVThoughtBubblePanel::NativeConstruct()
{
    Super::NativeConstruct();

    // Initialize with empty state
    ClearThought();
}

void UAegisAVThoughtBubblePanel::UpdateThought(const FAegisAgentThought& Thought)
{
    CurrentThought = Thought;

    // Update cognitive level
    if (CognitiveLevelText)
    {
        CognitiveLevelText->SetText(FText::FromString(Thought.CognitiveLevel));
    }

    // Update urgency with color coding
    if (UrgencyText)
    {
        UrgencyText->SetText(FText::FromString(Thought.Urgency));
        UrgencyText->SetColorAndOpacity(FSlateColor(GetUrgencyColor(Thought.Urgency)));
    }

    // Update goal
    if (GoalText)
    {
        GoalText->SetText(FText::FromString(Thought.CurrentGoal));
    }

    // Update target
    if (TargetText)
    {
        FString TargetDisplay = Thought.TargetAsset.IsEmpty() ? TEXT("None") : Thought.TargetAsset;
        TargetText->SetText(FText::FromString(TargetDisplay));
    }

    // Update situation
    if (SituationText)
    {
        SituationText->SetText(FText::FromString(Thought.Situation));
    }

    // Update considerations list
    UpdateConsiderationsDisplay();

    // Update options list
    UpdateOptionsDisplay();

    // Update decision
    if (DecisionText)
    {
        FString DecisionDisplay = Thought.SelectedDecision.IsEmpty() ? TEXT("Evaluating...") : Thought.SelectedDecision;
        DecisionText->SetText(FText::FromString(DecisionDisplay));
    }

    // Update confidence bar
    if (ConfidenceBar)
    {
        ConfidenceBar->SetPercent(Thought.Confidence);

        // Color based on confidence level
        FLinearColor Color;
        if (Thought.Confidence >= 0.8f)
        {
            Color = FLinearColor::Green;
        }
        else if (Thought.Confidence >= 0.5f)
        {
            Color = FLinearColor::Yellow;
        }
        else
        {
            Color = FLinearColor::Red;
        }
        ConfidenceBar->SetFillColorAndOpacity(Color);
    }

    // Update risk indicator
    if (RiskBar)
    {
        RiskBar->SetPercent(Thought.RiskScore);
        RiskBar->SetFillColorAndOpacity(GetRiskLevelColor(Thought.RiskLevel));
    }

    if (RiskLevelText)
    {
        FString RiskStr;
        switch (Thought.RiskLevel)
        {
        case ERiskLevel::Low: RiskStr = TEXT("LOW"); break;
        case ERiskLevel::Medium: RiskStr = TEXT("MEDIUM"); break;
        case ERiskLevel::High: RiskStr = TEXT("HIGH"); break;
        case ERiskLevel::Critical: RiskStr = TEXT("CRITICAL"); break;
        }
        RiskLevelText->SetText(FText::FromString(RiskStr));
        RiskLevelText->SetColorAndOpacity(FSlateColor(GetRiskLevelColor(Thought.RiskLevel)));
    }

    // Notify Blueprint
    OnThoughtUpdated(Thought);
}

void UAegisAVThoughtBubblePanel::ClearThought()
{
    CurrentThought = FAegisAgentThought();

    if (CognitiveLevelText) CognitiveLevelText->SetText(FText::FromString(TEXT("Idle")));
    if (UrgencyText) UrgencyText->SetText(FText::FromString(TEXT("None")));
    if (GoalText) GoalText->SetText(FText::FromString(TEXT("No active goal")));
    if (TargetText) TargetText->SetText(FText::FromString(TEXT("None")));
    if (SituationText) SituationText->SetText(FText::FromString(TEXT("Awaiting input...")));
    if (DecisionText) DecisionText->SetText(FText::FromString(TEXT("-")));
    if (ConfidenceBar) ConfidenceBar->SetPercent(0.0f);
    if (RiskBar) RiskBar->SetPercent(0.0f);
    if (RiskLevelText) RiskLevelText->SetText(FText::FromString(TEXT("LOW")));

    // Clear lists
    if (ConsiderationsBox) ConsiderationsBox->ClearChildren();
    if (OptionsBox) OptionsBox->ClearChildren();
}

void UAegisAVThoughtBubblePanel::UpdateConsiderationsDisplay()
{
    if (!ConsiderationsBox)
    {
        return;
    }

    ConsiderationsBox->ClearChildren();

    for (const FString& Consideration : CurrentThought.Considerations)
    {
        UTextBlock* ItemText = NewObject<UTextBlock>(ConsiderationsBox);
        ItemText->SetText(FText::FromString(FString::Printf(TEXT("- %s"), *Consideration)));
        ItemText->SetAutoWrapText(true);
        ConsiderationsBox->AddChildToVerticalBox(ItemText);
    }
}

void UAegisAVThoughtBubblePanel::UpdateOptionsDisplay()
{
    if (!OptionsBox)
    {
        return;
    }

    OptionsBox->ClearChildren();

    for (int32 i = 0; i < CurrentThought.Options.Num(); i++)
    {
        const FString& Option = CurrentThought.Options[i];
        UTextBlock* ItemText = NewObject<UTextBlock>(OptionsBox);
        ItemText->SetText(FText::FromString(FString::Printf(TEXT("%d. %s"), i + 1, *Option)));
        ItemText->SetAutoWrapText(true);
        OptionsBox->AddChildToVerticalBox(ItemText);
    }
}

FLinearColor UAegisAVThoughtBubblePanel::GetRiskLevelColor(ERiskLevel Level) const
{
    switch (Level)
    {
    case ERiskLevel::Low:
        return FLinearColor::Green;
    case ERiskLevel::Medium:
        return FLinearColor::Yellow;
    case ERiskLevel::High:
        return FLinearColor(1.0f, 0.5f, 0.0f); // Orange
    case ERiskLevel::Critical:
        return FLinearColor::Red;
    default:
        return FLinearColor::Gray;
    }
}

FLinearColor UAegisAVThoughtBubblePanel::GetUrgencyColor(const FString& Urgency) const
{
    if (Urgency.Equals(TEXT("critical"), ESearchCase::IgnoreCase))
    {
        return FLinearColor::Red;
    }
    else if (Urgency.Equals(TEXT("high"), ESearchCase::IgnoreCase))
    {
        return FLinearColor(1.0f, 0.5f, 0.0f); // Orange
    }
    else if (Urgency.Equals(TEXT("medium"), ESearchCase::IgnoreCase))
    {
        return FLinearColor::Yellow;
    }
    else
    {
        return FLinearColor::Green;
    }
}
