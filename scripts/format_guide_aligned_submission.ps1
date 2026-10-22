param(
    [string]$DocumentPath = 'C:\Users\HP\Desktop\Mabira_Mathew_R234371Y_Guide_Aligned.docx',
    [string]$PdfPath = 'C:\Users\HP\Desktop\Mabira_Mathew_R234371Y_Guide_Aligned.pdf'
)

$ErrorActionPreference = 'Stop'

$docPath = $DocumentPath
$pdfPath = $PdfPath
$wdSectionBreakNextPage = 2
$wdHeaderFooterPrimary = 1
$wdHeaderFooterFirstPage = 2
$wdHeaderFooterEvenPages = 3
$wdAlignPageNumberCenter = 1
$wdPageNumberStyleArabic = 0
$wdPageNumberStyleLowercaseRoman = 2

function Find-HeadingStart($document, $text) {
    for ($i = 1; $i -le $document.Paragraphs.Count; $i++) {
        $paragraph = $document.Paragraphs.Item($i)
        if ($paragraph.Range.Text.Trim() -eq $text) {
            return $paragraph.Range.Start
        }
    }
    throw "Could not find heading '$text'"
}

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($docPath, $false, $false)

    # Insert breaks from the later location first so earlier positions remain valid.
    $chapterStart = Find-HeadingStart $document 'CHAPTER 1: INTRODUCTION'
    $chapterRange = $document.Range($chapterStart, $chapterStart)
    $chapterRange.InsertBreak($wdSectionBreakNextPage)
    $declarationStart = Find-HeadingStart $document 'DECLARATION'
    $declarationRange = $document.Range($declarationStart, $declarationStart)
    $declarationRange.InsertBreak($wdSectionBreakNextPage)

    foreach ($section in $document.Sections) {
        $section.PageSetup.TopMargin = $word.CentimetersToPoints(2.54)
        $section.PageSetup.BottomMargin = $word.CentimetersToPoints(2.54)
        $section.PageSetup.LeftMargin = $word.CentimetersToPoints(2.54)
        $section.PageSetup.RightMargin = $word.CentimetersToPoints(2.54)
        $section.PageSetup.Gutter = 0
        $section.PageSetup.DifferentFirstPageHeaderFooter = $false
        $section.PageSetup.OddAndEvenPagesHeaderFooter = $false
        foreach ($index in @($wdHeaderFooterPrimary, $wdHeaderFooterFirstPage, $wdHeaderFooterEvenPages)) {
            $footer = $section.Footers.Item($index)
            $footer.LinkToPrevious = $false
            $footer.Range.Text = ''
        }
    }

    # Cover: no page number.  Preliminary pages: lower-case Roman.  Chapters: Arabic from 1.
    $prelim = $document.Sections.Item(2)
    $prelimFooter = $prelim.Footers.Item($wdHeaderFooterPrimary)
    $null = $prelimFooter.PageNumbers.Add($wdAlignPageNumberCenter, $true)
    $prelimFooter.PageNumbers.NumberStyle = $wdPageNumberStyleLowercaseRoman
    $prelimFooter.PageNumbers.RestartNumberingAtSection = $true
    $prelimFooter.PageNumbers.StartingNumber = 1

    $body = $document.Sections.Item(3)
    $bodyFooter = $body.Footers.Item($wdHeaderFooterPrimary)
    $null = $bodyFooter.PageNumbers.Add($wdAlignPageNumberCenter, $true)
    $bodyFooter.PageNumbers.NumberStyle = $wdPageNumberStyleArabic
    $bodyFooter.PageNumbers.RestartNumberingAtSection = $true
    $bodyFooter.PageNumbers.StartingNumber = 1

    foreach ($story in $document.StoryRanges) {
        try { $story.Fields.Update() | Out-Null } catch {}
    }
    $document.Fields.Update() | Out-Null
    $document.Save()
    $document.ExportAsFixedFormat($pdfPath, 17)
}
finally {
    if ($document) { $document.Close(0) }
    if ($word) { $word.Quit() }
}

Get-Item $docPath, $pdfPath | Select-Object FullName, Length, LastWriteTime
