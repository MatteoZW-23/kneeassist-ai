param(
    [string]$DocumentPath = "C:\Users\HP\Desktop\KneeAssist-AI\docs\academic\final_audit_20261018\deliverables\Mabira_Mathew_R234371Y_KneeAssist_XAI_Reconstructed_20261018.docx"
)

$ErrorActionPreference = 'Stop'
$PdfPath = [System.IO.Path]::ChangeExtension($DocumentPath, '.pdf')
$word = New-Object -ComObject Word.Application
$word.Visible = $false

try {
    $document = $word.Documents.Open($DocumentPath, $false, $false)

    # Covers: no visible page number.
    foreach ($index in 1, 2) {
        $footer = $document.Sections.Item($index).Footers.Item(1)
        $footer.LinkToPrevious = $false
        $footer.Range.Text = ''
    }

    # Preliminary material: lower-case Roman numerals from declaration onward.
    $preliminary = $document.Sections.Item(3).Footers.Item(1)
    $preliminary.LinkToPrevious = $false
    $preliminary.Range.Text = ''
    $preliminary.Range.ParagraphFormat.Alignment = 1 # centre
    $preliminary.PageNumbers.RestartNumberingAtSection = $true
    $preliminary.PageNumbers.StartingNumber = 1
    $preliminary.PageNumbers.NumberStyle = 2 # lower-case Roman
    $null = $preliminary.PageNumbers.Add(1, $true)

    # The fourth section is an internal transition. Keep it continuous so it
    # does not generate an unexplained blank page before Chapter 1.
    $document.Sections.Item(4).PageSetup.SectionStart = 0 # continuous
    $transition = $document.Sections.Item(4).Footers.Item(1)
    $transition.LinkToPrevious = $false
    $transition.Range.Text = ''

    # Chapter 1 and all remaining main matter: Arabic numbering from 1.
    $main = $document.Sections.Item(5).Footers.Item(1)
    $main.LinkToPrevious = $false
    $main.Range.Text = ''
    $main.Range.ParagraphFormat.Alignment = 1 # centre
    $main.PageNumbers.RestartNumberingAtSection = $true
    $main.PageNumbers.StartingNumber = 1
    $main.PageNumbers.NumberStyle = 0 # Arabic
    $null = $main.PageNumbers.Add(1, $true)

    $null = $document.Fields.Update()
    foreach ($section in $document.Sections) {
        $null = $section.Headers.Item(1).Range.Fields.Update()
        $null = $section.Footers.Item(1).Range.Fields.Update()
    }
    $document.Save()
    $document.ExportAsFixedFormat($PdfPath, 17)
    Write-Output "DOCX: $DocumentPath"
    Write-Output "PDF:  $PdfPath"
}
finally {
    if ($document) { $document.Close() }
    $word.Quit()
}
