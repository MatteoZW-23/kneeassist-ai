param(
    [Parameter(Mandatory = $true)] [string] $Path
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false

function Find-ParagraphStart($Document, [string] $Needle) {
    $range = $Document.Content.Duplicate
    $range.Find.ClearFormatting()
    $range.Find.Text = $Needle
    $range.Find.Forward = $true
    $range.Find.Wrap = 0
    if ($range.Find.Execute()) { return $range.Start }
    throw "Required heading not found: $Needle"
}

try {
    $doc = $word.Documents.Open($Path, $false, $false)
    if ($doc.Sections.Count -eq 1) {
        $chapterStart = Find-ParagraphStart $doc 'CHAPTER 1: INTRODUCTION'
        $declarationStart = Find-ParagraphStart $doc 'DECLARATION'
        # Insert from the end so the earlier location remains valid.
        $doc.Range($chapterStart, $chapterStart).InsertBreak(2) # wdSectionBreakNextPage
        $doc.Range($declarationStart, $declarationStart).InsertBreak(2)
    }
    if ($doc.Sections.Count -ne 3) {
        throw "Expected three document sections; found $($doc.Sections.Count)."
    }
    foreach ($section in $doc.Sections) {
        $section.PageSetup.DifferentFirstPageHeaderFooter = $false
    }
    $doc.PageSetup.OddAndEvenPagesHeaderFooter = $false

    # Title page: no displayed page number.
    $titleFooter = $doc.Sections.Item(1).Footers.Item(1)
    $titleFooter.LinkToPrevious = $false
    $titleFooter.Range.Text = ''

    # Preliminary pages: lower-case Roman numerals, restarting at i.
    $prelimFooter = $doc.Sections.Item(2).Footers.Item(1)
    $prelimFooter.LinkToPrevious = $false
    $prelimFooter.Range.Text = ''
    $prelimFooter.Range.ParagraphFormat.Alignment = 1 # wdAlignParagraphCenter
    $prelimNumber = $prelimFooter.PageNumbers.Add(1, $true) # centred
    $prelimFooter.PageNumbers.RestartNumberingAtSection = $true
    $prelimFooter.PageNumbers.StartingNumber = 1
    $prelimFooter.PageNumbers.NumberStyle = 2 # wdPageNumberStyleLowercaseRoman

    # Main text: Arabic numerals, restarting at 1.
    $mainFooter = $doc.Sections.Item(3).Footers.Item(1)
    $mainFooter.LinkToPrevious = $false
    $mainFooter.Range.Text = ''
    $mainFooter.Range.ParagraphFormat.Alignment = 1
    $mainNumber = $mainFooter.PageNumbers.Add(1, $true)
    $mainFooter.PageNumbers.RestartNumberingAtSection = $true
    $mainFooter.PageNumbers.StartingNumber = 1
    $mainFooter.PageNumbers.NumberStyle = 0 # wdPageNumberStyleArabic

    foreach ($story in $doc.StoryRanges) {
        try { $story.Fields.Update() | Out-Null } catch {}
    }
    $doc.Save()
    $doc.Close()
}
finally {
    $word.Quit()
}
