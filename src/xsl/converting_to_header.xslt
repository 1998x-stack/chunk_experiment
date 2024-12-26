<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
    <xsl:output method="html" indent="yes"/>
    
    <!-- Identity transform -->
    <xsl:template match="@* | node()">
        <xsl:copy>
            <xsl:apply-templates select="@* | node()"/>
        </xsl:copy>
    </xsl:template>
    
    <!-- Example transformation: convert <div class="header"> to <h2> -->
    <xsl:template match="div[@class='header']">
        <h2>
            <xsl:apply-templates/>
        </h2>
    </xsl:template>
</xsl:stylesheet>