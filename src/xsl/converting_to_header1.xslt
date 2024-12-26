<?xml version="1.0" encoding="UTF-8"?>
<!--
    converting_to_header.xslt

    Description:
    This XSLT stylesheet transforms specific HTML elements into header tags.
    For example, it converts <div class="header"> to <h2>, <span class="header"> to <h3>, etc.

    Usage:
    Apply this stylesheet to your HTML content using an XSLT processor.
    The transformed HTML will have the specified elements converted to headers,
    facilitating easier parsing and sectioning.

    Note:
    - Ensure that the XSLT processor you are using supports the version specified below.
    - Modify the transformation templates as needed to suit your HTML structure.
-->
<xsl:stylesheet version="1.0"
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:html="http://www.w3.org/1999/xhtml"
    exclude-result-prefixes="html">

    <!-- Identity Transform -->
    <!--
        Copies all elements and attributes as-is.
        This serves as the default behavior.
    -->
    <xsl:template match="@* | node()">
        <xsl:copy>
            <xsl:apply-templates select="@* | node()"/>
        </xsl:copy>
    </xsl:template>

    <!-- Example Transformation: Convert <div class="header"> to <h2> -->
    <!--
        Matches any <div> element with class attribute equal to 'header'
        and transforms it into an <h2> element, preserving its content.
    -->
    <xsl:template match="div[@class='header']">
        <h2>
            <xsl:apply-templates/>
        </h2>
    </xsl:template>

    <!-- Example Transformation: Convert <span class="header"> to <h3> -->
    <xsl:template match="span[@class='header']">
        <h3>
            <xsl:apply-templates/>
        </h3>
    </xsl:template>

    <!-- Add more transformation rules as needed -->
    <!-- For instance, converting <p class="header"> to <h4> -->
    <xsl:template match="p[@class='header']">
        <h4>
            <xsl:apply-templates/>
        </h4>
    </xsl:template>

    <!-- Optional: Remove specific attributes from transformed headers -->
    <!--
        For example, removing the 'class' attribute from the newly created headers.
    -->
    <xsl:template match="h2/@class | h3/@class | h4/@class">
        <!-- Do not copy the 'class' attribute -->
    </xsl:template>

    <!-- Optional: Add a default header if none exists -->
    <!--
        This template adds a default <h1> header at the beginning of the body if no headers are present.
    -->
    <xsl:template match="body">
        <xsl:copy>
            <!-- Check if there are any header elements -->
            <xsl:if test="not(.//h1 | .//h2 | .//h3 | .//h4 | .//h5 | .//h6)">
                <h1>Default Title</h1>
            </xsl:if>
            <xsl:apply-templates select="@* | node()"/>
        </xsl:copy>
    </xsl:template>

</xsl:stylesheet>